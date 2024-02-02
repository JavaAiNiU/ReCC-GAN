import argparse 
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
from Gan_User.FC4_net.Fc4NetLoader import Fc4NetLoader
from torch.nn.functional import normalize
from typing import Union
from dataset.settings import USE_CONFIDENCE_WEIGHTED_POOLING
import cv2
from dataset.settings import DEVICE
from FC4.Fullyc4 import FC4

from torch.cuda.amp import custom_bwd, custom_fwd




class DifferentiableClamp(torch.autograd.Function):
    """
    In the forward pass this operation behaves like torch.clamp.
    But in the backward pass its gradient is 1 everywhere, as if instead of clamp one had used the identity function.
    """

    @staticmethod
    @custom_fwd
    def forward(ctx, input, min, max):
        return input.clamp(min=min, max=max)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None


def dclamp(input, min, max):
    """
    Like torch.clamp, but with a constant 1-gradient.
    :param input: The input that is to be clamped.
    :param min: The minimum value of the output.
    :param max: The maximum value of the output.
    """
    return DifferentiableClamp.apply(input, min, max)

cam2rgb = np.array([
        1.8795, -1.0326, 0.1531,
        -0.2198, 1.7153, -0.4955,
        0.0069, -0.5150, 1.5081,]).reshape((3, 3))
cam2rgb = torch.tensor(cam2rgb).to(DEVICE).to(dtype=torch.float32)

# def linearize(img, black_lvl=2048, saturation_lvl=2**14-1):
#     """
#     :param saturation_lvl: 2**14-1 is a common value. Not all images
#                            have the same value.
#     """
#     return torch.clip((img - black_lvl)/(saturation_lvl - black_lvl), 0, 1)
def linearize(img, count,split,black_lvl=2048, saturation_lvl=2**14-1):
    """
    :param saturation_lvl: 2**14-1 is a common value. Not all images
                           have the same value.
    """
    if  split == "train" :
        if count <= 195 :
            black_lvl = 1024
        elif count >= 196 and count <= 345:
            black_lvl = 2048
        elif count >= 346 and count <= 492:
            black_lvl = 256
        elif count >= 493 and count <= 642:
            black_lvl = 0
        elif count >= 643 and count <= 798:
            black_lvl = 255
        elif count >= 799 and count <= 951:
            black_lvl = 143
        elif count >= 952 and count <= 1103:
            black_lvl = 0
        elif count >= 1104 and count <= 1304:
            black_lvl = 128
    elif split == "val":
        if count <= 32 :
            black_lvl = 1024
        elif count >= 33 and count <= 57:
            black_lvl = 2048
        elif count >= 58 and count <= 81:
            black_lvl = 256
        elif count >= 82 and count <= 106:
            black_lvl = 0
        elif count >= 107 and count <= 132:
            black_lvl = 255
        elif count >= 133 and count <= 157:
            black_lvl = 143
        elif count >= 158 and count <= 182:
            black_lvl = 0
        elif count >= 183 and count <= 215:
            black_lvl = 128
    return np.clip((img - black_lvl)/(saturation_lvl - black_lvl), 0, 1)

def freeze(model):
    for param in model.parameters():
        param.requires_grad = False



class Generator(torch.nn.Module):
    def __init__(self,FC4version: float = 1.1):
        super(Generator, self).__init__()
        # squeezenet = SqueezeNetLoader(squeezenet_version).load(pretrained=True)
        fc4net = Fc4NetLoader(FC4version).load(pretrained=False)
        # self.backbone = nn.Sequential(*list(squeezenet.children())[0][:12])
        self.backbone = nn.Sequential(*list(fc4net.children()))     
    
    def forward(self, x: torch.Tensor,img_raw1:torch.Tensor,fname,lable1:torch.Tensor) -> Union[tuple, torch.Tensor,bool]:
    # def forward(self, x: torch.Tensor) -> Union[tuple, torch.Tensor,bool]:
        out = self.backbone(x)
        img_raw = img_raw1.clone()
        lable = lable1.clone()
        Isjpg = 1
        
        if USE_CONFIDENCE_WEIGHTED_POOLING:
            # Per-patch color estimates (first 3 dimensions)
            rgb = normalize(out[:, :3, :, :], dim=1)

            # Confidence (last dimension)
            confidence = out[:, 3:4, :, :]

            # Confidence-weighted pooling
            pred = normalize(torch.sum(torch.sum(rgb * confidence, 2), 2), dim=1)


            return pred, rgb, confidence

        # Summation pooling
        pred1 = normalize(torch.sum(torch.sum(out, 2), 2), dim=1)
        pred = pred1.clone()
        C_imag_list = []
        gt_image_list = []
        
        
        for j in range(img_raw1.shape[0]):

            cam_wb = dclamp(img_raw[j]/pred[j], 1e-6, 1).to(dtype=torch.float32)
            cam_wb_gt = dclamp(img_raw[j]/lable[j], 1e-6, 1).to(dtype=torch.float32)

            if Isjpg == 0 :
                rgb = torch.matmul(cam_wb, cam2rgb.T)
                rgb_gt = torch.matmul(cam_wb_gt, cam2rgb.T)
                rgb = dclamp(rgb, 1e-6, 1)**(1/2.2)
                rgb_gt = dclamp(rgb_gt, 1e-6, 1)**(1/2.2)
                image_list= (rgb*255).to(dtype=torch.float32)
                gt_image = (rgb_gt*255).to(dtype=torch.float32)
            
                image_list[185:432,475:648,] = 0
                gt_image[185:432,475:648,] = 0
                
                image_list = image_list.permute(2,0,1)
                gt_image = gt_image.permute(2,0,1)
                C_imag_list.append(image_list)
                gt_image_list.append(gt_image)
            elif Isjpg == 1:
                image_list = cam_wb
                gt_image = cam_wb_gt
                image_list[185:432,475:648,] = 0
                gt_image[185:432,475:648,] = 0
                
                image_list = image_list.permute(2,0,1)
                gt_image = gt_image.permute(2,0,1)
                C_imag_list.append(image_list)
                gt_image_list.append(gt_image)    
            # elif Isjpg == 2:
                
        C_imag_list = torch.stack(C_imag_list)
        gt_image_list = torch.stack(gt_image_list)

        return pred,C_imag_list,gt_image_list


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels , 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, input):
        # Concatenate image and condition image by channels to produce input
        x = self.model(input)
        return x
    
class ModelFC4:

    def __init__(self):
        self.__device = DEVICE
        self.__optimizer = None
        self.__network = FC4().to(self.__device)

    def predict(self, img: torch.Tensor, return_steps: bool = False) -> Union[torch.Tensor, tuple]:
        """
        Performs inference on the input image using the FC4 method.
        @param img: the image for which a colour of the illuminant has to be estimated
        @param return_steps: whether or not to also return the per-patch estimates and confidence weights. When this
        flag is set to True, confidence-weighted pooling must be active)
        @return: the colour estimate as a Tensor. If "return_steps" is set to true, the per-path colour estimates and
        the confidence weights are also returned (used for visualizations)
        """
        if USE_CONFIDENCE_WEIGHTED_POOLING:
            pred, rgb, confidence = self.__network(img)
            if return_steps:
                return pred, rgb, confidence
            return pred
        return self.__network(img)

class DecomNet(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(DecomNet, self).__init__()
        # Shallow feature extraction
        self.net1_conv0 = nn.Conv2d(4, channel, kernel_size * 3,
                                    padding=4, padding_mode='replicate')
        
        # Activated layers!
        self.net1_convs = nn.Sequential(nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU())
        # Final recon layer
        self.net1_recon = nn.Conv2d(channel, 4, kernel_size,
                                    padding=1, padding_mode='replicate')
        freeze(self)

    def forward(self, input_im):
        
        input_im = input_im.permute(0,3,1,2)
        input_max= torch.max(input_im, dim=1, keepdim=True)[0]
        input_img= torch.cat((input_max, input_im), dim=1)
        input_img   = self.net1_conv0(input_img)
        input_img   = self.net1_convs(input_img)
        input_img     = self.net1_recon(input_img)
        R        = torch.sigmoid(input_img[:, 0:3, :, :])
        L        = torch.sigmoid(input_img[:, 3:4, :, :])   
        return R,L
    
class ColorCoreect(nn.Module):
    def __init__(self):
        super(ColorCoreect, self).__init__()

        self.DecomNet  = DecomNet()
        model_statedict_D = torch.load("/home/sby/ColorConstancy/RetinexNet_PyTorch-master/ckpts/Decom/9200.tar",map_location=lambda storage,loc:storage)
        self.DecomNet.load_state_dict(model_statedict_D)
        print("model.fc1.weight", self.DecomNet.net1_conv0.weight)
        self.Generator= Generator()
    def forward(self, img_raw1:torch.Tensor,fname,lable1:torch.Tensor):
        # input_low = Variable(torch.FloatTensor(torch.from_numpy(img_raw1))).cuda()
        R_low, I_low   = self.DecomNet(img_raw1)
        pred,C_imag_list,gt_image_list = self.Generator(R_low,img_raw1,fname,lable1)
        return pred,C_imag_list,gt_image_list
    
