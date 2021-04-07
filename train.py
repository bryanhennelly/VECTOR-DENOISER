import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange
import scipy.signal
from tensorboardX import SummaryWriter
import time
import pdb
import argparse

# self defined modules
from models import CAE
import utils

def loss_function_peaks(recon_x, x, sw): #hidden_neurons_batch: [samples, number of neurons]  ##########################################################################################
    # BCE = F.mse_loss(recon_x.view(-1, 1000), x.view(-1, 1000))
    BCE = F.l1_loss(recon_x.view(-1, 1000)*sw.view(-1,1000), x.view(-1, 1000)*sw.view(-1,1000)) #####################################################################
    #print((torch.sum(sw.view(-1,1000))))
    ## 
    return BCE.cuda()
    
def loss_function_global(recon_x, x): #hidden_neurons_batch: [samples, number of neurons]  ##########################################################################################
    # BCE = F.mse_loss(recon_x.view(-1, 1000), x.view(-1, 1000))
    BCE = F.l1_loss(recon_x.view(-1, 1000), x.view(-1, 1000)) #####################################################################
    return BCE.cuda()   

 

def adjust_learning_rate(learning_rate, optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr_new = learning_rate * (0.1 ** (sum(epoch >= np.array(lr_steps))))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_new


# windows might need to manually change in the function
def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_id', type=str, default='0')
    parser.add_argument('--is_train', default=False, action='store_true')
    parser.add_argument('--is_skip', default=False, action='store_true')
    parser.add_argument('--batch_size', '--bs', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--lr_steps', type=float, default=[5,10,15,20], nargs="+",
                        help='lr steps for decreasing learning rate') # batch: 64 [10,20,30], [5,10,20]
    parser.add_argument('--base_model', default='cae_4', type=str)
    parser.add_argument('--dataset', default=1, type=int)
    parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--epochs', default=25, type=int, metavar='N',
                    help='number of total epochs to run')
    parser.add_argument('--alpha', default=5, type=float)
    args = parser.parse_args()
    return args

args = parse_opts()
alpha = args.alpha/10
print(alpha)

print(args)


os.environ['CUDA_VISIBLE_DEVICES']=args.cuda_id

# create tensorboard writer
cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))



if args.base_model == 'cae_4':
    model = CAE.CAE_4(data_len=1000, kernel_size=8, is_skip=args.is_skip)
elif args.base_model == 'cae_5':
    model = CAE.CAE_5(data_len=1000, kernel_size=8, is_skip=args.is_skip)
elif args.base_model == 'cae_6':
    model = CAE.CAE_6(data_len=1000, kernel_size=8, is_skip=args.is_skip)
elif args.base_model == 'cae_7':
    model = CAE.CAE_7(data_len=1000, kernel_size=8, is_skip=args.is_skip)
elif args.base_model == 'cae_8':
    model = CAE.CAE_8(data_len=1000, kernel_size=8, is_skip=args.is_skip)
elif args.base_model == 'cae_9':
    model = CAE.CAE_9(data_len=1000, kernel_size=8, is_skip=args.is_skip)
model.cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

from Generate_Data import *
class raman_dataset_fast(Dataset):
    def __init__(self, dataset, size):
        self.cars_data, self.raman_data, self.sw_data = generate_datasets(dataset,size) ##############################################################################################
         
    def __len__(self):
        return len(self.raman_data)

    def __getitem__(self, idx):
        raman_data = self.raman_data[idx]
        cars_data = self.cars_data[idx]
        ##############################################################################################################################################################################
        sw_data = self.sw_data[idx]
        ##############################################################################################################################################################################
        return raman_data, cars_data, sw_data ################################################################################################################################################

class raman_dataset(Dataset):
    def __init__(self, file_path, raman_file, cars_file, sw_file):
        self.raman_data = pd.read_csv(os.path.join(file_path, raman_file)).iloc[:, 1:]
        self.cars_data = pd.read_csv(os.path.join(file_path, cars_file)).iloc[:, 1:]
        self.sw_data = pd.read_csv(os.path.join(file_path, sw_file)).iloc[:, 1:]

        
    def __len__(self):
        return len(self.raman_data)

    def __getitem__(self, idx):
        raman_data = self.raman_data.values[idx]
        cars_data = self.cars_data.values[idx]
        sw_data = self.sw_data.values[idx]
        return raman_data, cars_data, sw_data

# define model save path
if args.is_skip == True:
    model_save_dir = os.path.join('trained_model', '{}-skip'.format(args.base_model),'{}-dataset'.format(args.dataset))
    logdir = os.path.join('log', '{}-skip'.format(args.base_model),'{}-dataset'.format(args.dataset))
else:
    model_save_dir = os.path.join('trained_model', '{}-noskip'.format(args.base_model),'{}-dataset'.format(args.dataset))
    logdir = os.path.join('log', '{}-noskip'.format(args.base_model),'{}-dataset'.format(args.dataset))

print('Before if is train')

# training
if args.is_train:
    print(alpha)
    print('Loading dataset.....')
    dataset_train = raman_dataset_fast(args.dataset,100000)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    dataset_val = raman_dataset_fast(args.dataset,20000)
    val_loader = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    writer = SummaryWriter(log_dir=logdir)

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)


    # training
    best_loss = 100. # check if best val loss to save model
    train_loss = utils.AverageMeter() # train loss
    val_loss = utils.AverageMeter() # validation loss
    train_loss_peaks = utils.AverageMeter() # train loss
    val_loss_peaks = utils.AverageMeter() # validation loss
    train_loss_global = utils.AverageMeter() # train loss
    val_loss_global = utils.AverageMeter() # validation loss
    for epoch in trange(args.epochs):
        model.train()
        train_loss.reset()
        val_loss.reset()
        train_loss_peaks.reset()
        val_loss_peaks.reset()        
        train_loss_global.reset()
        val_loss_global.reset()

        for step, inputs in enumerate(train_loader):
            raman = inputs[0].float().cuda()
            cars = inputs[1].float().cuda()
            #############################################################################################################################################################################
            sw = inputs[2].float().cuda()
            ############################################################################################################################################################################
            optimizer.zero_grad()
            outputs = model(cars)
            loss_peaks = loss_function_peaks(outputs, raman, sw) #############################################################################################################################
            loss_global = loss_function_global(outputs, raman) ######################################################
            loss = alpha*loss_peaks + (1-alpha)*loss_global ####################################################################
            loss.backward()
            optimizer.step()
            train_loss.update(loss.item(), raman.size(0))
            train_loss_peaks.update(loss_peaks.item(), raman.size(0))
            train_loss_global.update(loss_global.item(), raman.size(0))


            if (step+1) % 20 == 0:
                print_string = ('Epoch: [{0}][{1}/{2}], '
                                 'lr: {lr:.5f}, '
                                 'Train loss: {loss.val:.5f} ({loss.avg:.5f})'
                                 .format(epoch, step, len(train_loader), lr=optimizer.param_groups[0]['lr'], loss=train_loss)
                )
                print(print_string)
                print_string = ('Epoch: [{0}][{1}/{2}], '
                                 'lr: {lr:.5f}, '
                                 'Train loss peaks: {loss.val:.5f} ({loss.avg:.5f})'
                                 .format(epoch, step, len(train_loader), lr=optimizer.param_groups[0]['lr'], loss=train_loss_peaks)
                )
                print(print_string)
                print_string = ('Epoch: [{0}][{1}/{2}], '
                                 'lr: {lr:.5f}, '
                                 'Train loss global {loss.val:.5f} ({loss.avg:.5f})'
                                 .format(epoch, step, len(train_loader), lr=optimizer.param_groups[0]['lr'], loss=train_loss_global)
                )
                print(print_string)
                
           
            
        # validation
        model.eval()
        with torch.no_grad():
            for val_step, inputs in enumerate(val_loader):
                raman = inputs[0].float().cuda()
                cars = inputs[1].float().cuda()
                #############################################################################################################################################################################
                sw = inputs[2].float().cuda()
                ############################################################################################################################################################################
                outputs = model(cars)
                loss_valid_peaks = loss_function_peaks(outputs, raman, sw) #############################################################################################################################
                loss_valid_global = loss_function_global(outputs, raman)
                loss_valid = alpha*loss_valid_peaks+(1-alpha)*loss_valid_global ######################################################################################################################
                val_loss.update(loss_valid.item(), raman.size(0))
                val_loss_peaks.update(loss_valid_peaks.item(), raman.size(0))
                val_loss_global.update(loss_valid_global.item(), raman.size(0))
        print_string = ('Test: [{0}], '
                        'Val loss: {loss.avg:.5f}'
                        
                        .format(len(val_loader), loss=val_loss)                        
        )
        print(print_string)
        print_string = ('Test: [{0}], '
                        'Val loss peaks: {loss.avg:.5f}'
                        
                        .format(len(val_loader), loss=val_loss_peaks)                        
        )
        print(print_string)
        print_string = ('Test: [{0}], '
                        'Val loss global: {loss.avg:.5f}'
                        
                        .format(len(val_loader), loss=val_loss_global)                        
        )
        print(print_string)
        
        # adjust learning rate 
        adjust_learning_rate(args.lr, optimizer, epoch, args.lr_steps)
        writer.add_scalar('train_loss', train_loss.avg, epoch)
        writer.add_scalar('val_loss', val_loss.avg, epoch)
        writer.add_scalar('train_loss_peaks', train_loss_peaks.avg, epoch)
        writer.add_scalar('val_loss_peaks', val_loss_peaks.avg, epoch)
        writer.add_scalar('train_loss_global', train_loss_global.avg, epoch)
        writer.add_scalar('val_loss_global', val_loss_global.avg, epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        # save the best model
        if val_loss.avg < best_loss:
            checkpoint = os.path.join(model_save_dir, 'checkpoint'+str(args.dataset)+'_alpha_0_'+str(int(args.alpha))+'.pth.tar')
            utils.save_checkpoint(model, optimizer, epoch, checkpoint)        
            best_loss = val_loss.avg
        print('Best loss: {:.5f}'.format(best_loss))
    print('Finished Training/Validation')
else: # testing
    print('Loading dataset.....')
    if args.dataset == 1:
        a=1
        b='a'
    elif args.dataset == 2:
        a=1
        b='b'
    elif args.dataset == 3:
        a=1
        b='c'
    elif args.dataset == 4:
        a=2
        b='a'
    elif args.dataset == 5:
        a=2
        b='b'
    elif args.dataset == 6:
        a=2
        b='c'
    elif args.dataset == 7:
        a=3
        b='a'
    elif args.dataset == 8:
        a=3
        b='b'
    else:
        a=3
        b='c'
    dataset_val = raman_dataset('data', str(a)+b+'Raman.csv', str(a)+b+'CARS.csv', str(a)+b+'SW.csv')
    val_loader = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    checkpoint_path = os.path.join(model_save_dir, 'checkpoint'+str(args.dataset)+'_alpha_0_'+str(int(args.alpha))+'.pth.tar')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.cuda()
    model.eval()
    val_loss = utils.AverageMeter() # validation loss
    val_loss_peaks = utils.AverageMeter() # validation loss
    val_loss_global = utils.AverageMeter() # validation loss
  
    
    with torch.no_grad():
        results=[]
        for val_step, inputs in enumerate(tqdm(val_loader)):
            raman = inputs[0].float().cuda()
            cars = inputs[1].float().cuda()
            sw = inputs[2].float().cuda()
            outputs = model(cars)
            results.append((outputs.cpu()).numpy())

            loss_valid_peaks = loss_function_peaks(outputs, raman, sw) #############################################################################################################################
            loss_valid_global = loss_function_global(outputs, raman)
            loss_valid = alpha*loss_valid_peaks+(1-alpha)*loss_valid_global ######################################################################################################################
            val_loss.update(loss_valid.item(), raman.size(0))
            val_loss_peaks.update(loss_valid_peaks.item(), raman.size(0))
            val_loss_global.update(loss_valid_global.item(), raman.size(0))
 

            #loss_valid = loss_function_global(outputs, raman)
            #val_loss.update(loss_valid.item(), raman.size(0))
        print(np.size(results))
        results = np.array(results)
        results = results.reshape(results.shape[1],results.shape[2])
        print(np.size(results))
        pd.DataFrame(results).to_csv('./data_val/'+str(a)+b+'Raman_spectrums_results.csv')
    print_string = 'Test: loss: {loss:.5f}'.format(loss=val_loss.avg)
    print(print_string)
    print_string = 'Test: loss peaks: {loss:.5f}'.format(loss=val_loss_peaks.avg)
    print(print_string)
    print_string = 'Test: Global loss: {loss:.5f}'.format(loss=val_loss_global.avg)
    print(print_string)
 
 
 
 
 
 
 
 
