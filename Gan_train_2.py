import os, argparse
import time
import sys
sys.path.append('/home/sby/ColorConstancy/')

import torch
from torch.utils.data import DataLoader

from dataset.settings import DEVICE, USE_CONFIDENCE_WEIGHTED_POOLING
from dataset.commeen import print_metrics, log_metrics
# from SHEJI.SLSMIDataset import get_loader
from Gan_User.Date.Gan_dataset_resize import get_loader
from FC4.ModelFC4 import ModelFC4
from FC4.training.Evaluator import Evaluator
from FC4.training.LossTracker import LossTracker,GANLoss
from torch.utils.tensorboard import SummaryWriter
from ModelGan_gt import ModelGenerator,ModelDiscriminator,ModelSwinTransformer
from YZheng_model.code.ModelGan_YZ.Model_ResNet import ModelResNet
from torch.autograd import Variable


import numpy as np
import torch.optim as optim
import cv2
import rawpy
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


torch.autograd.set_detect_anomaly(True)

cam2rgb = np.array([
        1.8795, -1.0326, 0.1531,
        -0.2198, 1.7153, -0.4955,
        0.0069, -0.5150, 1.5081,]).reshape((3, 3))
cam2rgb = torch.tensor(cam2rgb)

EPOCHS = 300
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
LEARNING_RATE_D = 1e-8

FOLD_NUM = 0

RELOAD_CHECKPOINT = False 
#If there is a pre-trained model, give its model file path here, G is the generator and D is the discriminator
#如果有预训练模型，在这里给他的路径，G是生成器，D是判别器
PATH_TO_PTH_CHECKPOINT_G = ("")
PATH_TO_PTH_CHECKPOINT_D = ("")


def main(config):
    #Path to save logs
    #保存日志的路径
    path_to_log_D = os.path.join("/home/sby/ColorConstancy/Gan_User/logs_D", "fold_{}_{}".format(str(FOLD_NUM), str(time.time())))
    path_to_log_G = os.path.join("/home/sby/ColorConstancy/Gan_User/logs_G", "fold_{}_{}".format(str(FOLD_NUM), str(time.time())))
    os.makedirs(path_to_log_D, exist_ok=True)
    os.makedirs(path_to_log_G, exist_ok=True)
    path_to_metrics_log_D = os.path.join(path_to_log_D, "metrics_D.csv")
    path_to_metrics_log_G = os.path.join(path_to_log_G, "metrics_G.csv")
    
    print('===> Building models')
    # Model_G = ModelGenerator()
    Model_G = ModelResNet()
    
    Model_D = ModelDiscriminator()
    
    if RELOAD_CHECKPOINT:
        print('\n Reloading checkpoint - pretrained model stored at: {} \n'.format(PATH_TO_PTH_CHECKPOINT_G))
        print('\n Reloading checkpoint - pretrained model stored at: {} \n'.format(PATH_TO_PTH_CHECKPOINT_D))
        Model_G.load(PATH_TO_PTH_CHECKPOINT_G)
        Model_D.load(PATH_TO_PTH_CHECKPOINT_D)
        
    

    Model_G.print_network()
    Model_G.log_network(path_to_log_G)
    Model_G.set_optimizer(LEARNING_RATE)

    Model_D.log_network(path_to_log_D)
    Model_D.set_optimizer(LEARNING_RATE_D)

    train_loader = get_loader(config, 'train')
    val_loader = get_loader(config, 'val')
    
    
    print("\n**************************************************************")
    print("\t\t\t Training Gan - Fold {}".format(FOLD_NUM))
    print("**************************************************************\n")
    
    evaluator = Evaluator()
    best_val_loss, best_metrics ,best_val_loss_1= 100.0, evaluator.get_best_metrics(),100.0
    train_loss_G, train_loss_Gan,train_loss_l1,train_loss_l2 = LossTracker(), LossTracker(),LossTracker(),LossTracker(),
    val_loss_G,val_loss_Gan,val_loss_l1,val_loss_l2 = LossTracker(),LossTracker(),LossTracker(),LossTracker()
    train_loss_D =  GANLoss().to(DEVICE)
    criterionL1 = nn.L1Loss().to(DEVICE)
    loss_d = 0
    #The path stored by tensorboard is defined as your own path
    #tensorboard存储的路径，定义成自己的路径
    writer = SummaryWriter(log_dir="/home/sby/ColorConstancy/Gan_User/checkpoint/tubiao/new_Resnet")
    writer1 = SummaryWriter(log_dir="/home/sby/ColorConstancy/Gan_User/checkpoint/tubiao/new_Resnet")
    for epoch in range(93,EPOCHS + 1 ):
        Model_G.train_mode()
        Model_D.train_mode()
        train_loss_G.reset()
        train_loss_D.reset()
        train_loss_Gan.reset()
        train_loss_l1.reset()
        train_loss_l2.reset()
        val_loss_G.reset()
        val_loss_Gan.reset()
        val_loss_l1.reset()
        val_loss_l2.reset()
        start = time.time()
        if epoch == 40 :
            Model_G.set_optimizer(1e-6)
            Model_D.set_optimizer(1e-9)
        if epoch == 80:
            Model_G.set_optimizer(1e-7) 
            Model_D.set_optimizer(1e-10)
        
        for i, (img,img_raw,gt_img, label,fname ) in enumerate(train_loader): 
            img, label,img_raw, gt_img = img.to(DEVICE), label.to(DEVICE),img_raw.to(DEVICE),gt_img.to(DEVICE)
                  
            ######
            Model_D.reset_gradient()    
            Model_G.evaluation_mode()
            Model_D.train_mode()
            ######
        
            img_raw_per = img_raw.permute(0,3,1,2)
            pred,C_imag_list,gt_img_list = Model_G.predict(img,img_raw,fname,label)
            
            ######################
            # (1) Update D network
            ######################
            
            # train with fake
            if epoch % 5 == 0:
                # fake_ab = torch.cat((img_raw_per, C_imag_list), 1)
                fake_ab =  C_imag_list
                pred_fake = Model_D.predict(fake_ab.detach())
                loss_d_fake = train_loss_D(pred_fake, False)
            
            
            # train with real
                # real_ab = torch.cat((img_raw_per, gt_img_list), 1)
                real_ab =  gt_img
                pred_real = Model_D.predict(real_ab)
                loss_d_real = train_loss_D(pred_real, True)
            
            # Combined D loss
                loss_d = (loss_d_fake + loss_d_real) * 0.5

                loss_d.backward()
            # optimizer_d.step()
                # if(loss_d.data > 0.4):
                Model_D.optimize()
                train_loss_D.update(loss_d)
            

            #### 
            ######################
            # (2) Update G network
            ######################

            Model_G.reset_gradient()
            ######
            Model_G.train_mode()
            Model_D.evaluation_mode()
            
            pred,C_imag_list,gt_img_list = Model_G.predict(img,img_raw,fname,label)
            ######
            img_raw = img_raw.permute(0,3,1,2)
            # First, G(A) should fake the discriminator
            fake_ab =  C_imag_list.clone()
            pred_fake = Model_D.predict(fake_ab.detach())
            loss_g_gan = train_loss_D(pred_fake, True)#1.1031
            
            
            # Second, G(A) = B
            loss_g_2 = criterionL1(C_imag_list.detach(), gt_img.detach()) * 0.01#7.3571          
            loss_g_1 = Model_G.optimize_loss(pred,label)#25.4016
            loss_g =  loss_g_1 + loss_g_gan + loss_g_2#33.8618
            
            
            loss_g.backward() 
            Model_G.optimize()
            train_loss_G.update(loss_g)
            train_loss_Gan.update(loss_g_gan)
            train_loss_l1.update(loss_g_1)
            train_loss_l2.update(loss_g_2)
            
            if i % 5 == 0:
                print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f} Loss_G_gan :{:.4f} Loss_G_l1:{:.4f} Loss_G_l2:{:.4f} ".format(
                    epoch,EPOCHS ,i, loss_d, train_loss_G.avg,train_loss_Gan.avg,train_loss_l1.avg,train_loss_l2.avg)) 
               
        if epoch % 1 == 0:
                    y = train_loss_l1.avg.detach().cpu().numpy()
                    z = train_loss_l2.avg.detach().cpu().numpy()
                    q = train_loss_G.avg.detach().cpu().numpy()
                    x = train_loss_Gan.avg.detach().cpu().numpy()

                    writer.add_scalar("train/Loss_G_gan", x, epoch)
                    writer.add_scalar("train/train_loss_l1", y, epoch)
                    writer.add_scalar("train/train_loss_l2", z, epoch)
                    writer.add_scalar("train/train_loss_g", q, epoch)

        if epoch % 5 == 0:
                    w = train_loss_D.avg.detach().cpu().numpy()
                    writer.add_scalar("train/train_loss_d", w, epoch)
            
        
              
        train_time = time.time() - start
        
        start = time.time()

        
        if epoch % 2 == 0:
            evaluator.reset_errors()
            Model_G.evaluation_mode()  
            
            print("\n--------------------------------------------------------------")
            print("\t\t\t Validation")
            print("--------------------------------------------------------------\n") 
            
            with torch.no_grad():
                for i, (img,img_raw,gt_img, label,fname ) in enumerate(val_loader):
                    img_raw, label,img,gt_img = img_raw.to(DEVICE), label.to(DEVICE),img.to(DEVICE),gt_img.to(DEVICE)

                    pred,C_imag_list,gt_img_list = Model_G.predict(img,img_raw,fname,label)
                    img_raw = img_raw.permute(0,3,1,2)
                    fake_ab =  C_imag_list.clone()
                    pred_fake = Model_D.predict(fake_ab.detach())
                    loss_g_gan = train_loss_D(pred_fake, True)
                    loss_g_2 = criterionL1(C_imag_list.detach(), gt_img.detach())*0.01          
                    loss_g_1 = Model_G.optimize_loss(pred,label)
                    v_loss_g =  loss_g_1 + loss_g_gan + loss_g_2
                    
                    val_loss_G.update(v_loss_g)
                    val_loss_Gan.update(loss_g_gan)
                    val_loss_l1.update(loss_g_1)
                    val_loss_l2.update(loss_g_2)
                    
                    evaluator.add_error(v_loss_g.detach().cpu().numpy())

                    if i % 2 == 0:
                        print("[ Epoch: {}/{} - Batch: {}] | Val loss: {:.4f} ]".format(epoch, EPOCHS, i, val_loss_G.avg)) 
 
                    
                if epoch % 1 == 0:
                    A = val_loss_l1.avg.detach().cpu().numpy()
                    B = val_loss_l2.avg.detach().cpu().numpy()
                    C = val_loss_G.avg.detach().cpu().numpy()
                    D = val_loss_Gan.avg.detach().cpu().numpy()                 
                    
                    writer1.add_scalar("val/loss_l1", A, epoch)
                    writer1.add_scalar("val/loss_l2", B, epoch)
                    writer1.add_scalar("val/loss_g_gan", D, epoch) 
                    writer1.add_scalar("val/loss_g", C, epoch)   
                                    
            print("\n--------------------------------------------------------------\n")
        val_time = time.time() - start
        
        # metrics = evaluator.compute_metrics()   
        print("\n********************************************************************")
        print(" Train Time ... : {:.4f}".format(train_time))
        print(" Train Loss ... : {:.4f}".format(train_loss_G.avg))    

        if val_time > 0.1:
            print("....................................................................")
            print(" Val Time ..... : {:.4f}".format(val_time)) 
            print(" Val_l1 loss ..... : {:.4f}".format(val_loss_l1.avg))
            print(" Val_l2 loss ..... : {:.4f}".format(val_loss_l2.avg))
            print(" Val_Gan loss ..... : {:.4f}".format(val_loss_Gan.avg))
            print(" Val loss ..... : {:.4f}".format(val_loss_G.avg))
            print("....................................................................")
        print("********************************************************************\n")
    
        if epoch % 5 == 0:
            if not os.path.exists("checkpoint"):
                os.mkdir("checkpoint")
            if not os.path.exists(os.path.join("checkpoint", config.model_root)):
                os.mkdir(os.path.join("checkpoint", config.dataset))
            
            print("Checkpoint saved to {}".format("checkpoint" + config.model_root))
        if 0 < val_loss_G.avg < best_val_loss:
            model_g_model_out_path_best = "{}checkpoint/best/new_Resnet/netG_model_epoch_{}_loss_{}.pth".format(config.model_root, epoch,val_loss_G.avg)
            model_d_model_out_path_best = "{}checkpoint/best/new_Resnet/netD_model_epoch_{}_loss_{}.pth".format(config.model_root, epoch,val_loss_G.avg)
            best_val_loss = val_loss_G.avg

            print("Saving new best model... \n")
            Model_G.save( model_g_model_out_path_best )
            Model_D.save( model_d_model_out_path_best )
        if 0 < val_loss_l1.avg < best_val_loss_1:
            model_g_model_out_path_best = "{}checkpoint/val_1/new_Resnet/netG_model_epoch_{}_loss_{}.pth".format(config.model_root, epoch,val_loss_l1.avg)
            model_d_model_out_path_best = "{}checkpoint/val_1/new_Resnet/netD_model_epoch_{}_loss_{}.pth".format(config.model_root, epoch,val_loss_l1.avg)
            best_val_loss_1 = val_loss_l1.avg
            # best_metrics = evaluator.update_best_metrics()
            print("Saving new val_1.best model... \n")
            Model_G.save( model_g_model_out_path_best )
            Model_D.save( model_d_model_out_path_best )   

    writer.close()
    writer1.close()
        


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    writer = SummaryWriter(log_dir="summary_pic")

    # training hyper-parameters
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)

    # dataset & loader config
    parser.add_argument('--image_pool', type=str, nargs='+', default=['1'])
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--input_type', type=str, default='rgb', choices=['rgb','uvl'])
    parser.add_argument('--output_type', type=str, default=None, choices=['illumination','uv','mixmap'])
    parser.add_argument('--mask_black', type=str, default=None)
    parser.add_argument('--mask_highlight', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=10)

    # path config
    parser.add_argument('--data_root', type=str, default='/raid/sby/CubeAllData/Cube++/new_data')
    parser.add_argument('--model_root', type=str, default='/home/sby/ColorConstancy/Gan_User/')
    parser.add_argument('--result_root', type=str, default='results')
    parser.add_argument('--log_root', type=str, default='logs')
    parser.add_argument('--checkpoint', type=str, default='210520_0600')

    # Misc
    parser.add_argument('--save_epoch', type=int, default=-1,
                        help='number of epoch for auto saving, -1 for turn off')
    parser.add_argument('--multi_gpu', type=int, default=1, choices=[0,1,2,3],
                        help='0 for single-GPU, 1 for multi-GPU')
    parser.add_argument('--lamb', type=int, default=0.1, help='weight on L1 term in objective')

    config = parser.parse_args()
    main(config)          
            
