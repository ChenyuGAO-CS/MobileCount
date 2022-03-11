import numpy as np
import os
import math
import time
import random
import shutil
from tqdm import tqdm

import torch
from torch import nn


import torchvision.utils as vutils
import torchvision.transforms as standard_transforms

import pdb


def initialize_weights(models):
    for model in models:
        real_init_weights(model)


def real_init_weights(m):

    if isinstance(m, list):
        for mini_m in m:
            real_init_weights(mini_m)
    else:
        if isinstance(m, nn.Conv2d):    
            nn.init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0.0, std=0.01)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m,nn.Module):
            for mini_m in m.children():
                real_init_weights(mini_m)
        else:
            print( m )

def weights_normal_init(*models):
    for model in models:
        dev=0.01
        if isinstance(model, list):
            for m in model:
                weights_normal_init(m, dev)
        else:
            for m in model.modules():            
                if isinstance(m, nn.Conv2d):        
                    m.weight.data.normal_(0.0, dev)
                    if m.bias is not None:
                        m.bias.data.fill_(0.0)
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0.0, dev)


def logger(exp_path, exp_name, work_dir, exception, resume=False):

    from tensorboardX import SummaryWriter
    
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)
    writer = SummaryWriter(exp_path+ '/' + exp_name)
    log_file = exp_path + '/' + exp_name + '/' + exp_name + '.txt'
    
    cfg_file = open('./config.py',"r")  
    cfg_lines = cfg_file.readlines()
    
    with open(log_file, 'a') as f:
        f.write(''.join(cfg_lines) + '\n\n\n\n')

    if not resume:
        copy_cur_env(work_dir, os.path.join(exp_path,exp_name,'code'), exception)


    return writer, log_file



def logger_for_CMTL(exp_path, exp_name, work_dir, exception, resume=False):
    
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    if not os.path.exists(exp_path+ '/' + exp_name):
        os.mkdir(exp_path+ '/' + exp_name)
    log_file = exp_path + '/' + exp_name + '/' + exp_name + '.txt'
    
    cfg_file = open('./config.py',"r")  
    cfg_lines = cfg_file.readlines()
    
    with open(log_file, 'a') as f:
        f.write(''.join(cfg_lines) + '\n\n\n\n')

    if not resume:
        copy_cur_env(work_dir, exp_path+ '/' + exp_name + '/code', exception)


    return log_file

def logger_txt(log_file,epoch,scores):

    mae, mse, mgape, loss = scores

    snapshot_name = 'all_ep_%d_mae_%.1f_rmse_%.1f_mgape_%.1f' % (epoch + 1, mae, mse, mgape)

    # pdb.set_trace()

    with open(log_file, 'a') as f:
        f.write('='*15 + '+'*15 + '='*15 + '\n\n')
        f.write(snapshot_name + '\n')
        f.write('    [mae %.2f rmse %.2f mgape %.2f], [val loss %.4f]\n' % (mae, mse, mgape, loss))
        f.write('='*15 + '+'*15 + '='*15 + '\n\n')    





def vis_results(exp_name, epoch, writer, restore, img, pred_map, gt_map):

    pil_to_tensor = standard_transforms.ToTensor()

    x = []
    
    for idx, tensor in enumerate(zip(img.cpu().data, pred_map, gt_map)):
        if idx>1:# show only one group
            break
        pil_input = restore(tensor[0])
        pil_output = torch.from_numpy(tensor[1]/(tensor[2].max()+1e-10)).repeat(3,1,1)
        pil_label = torch.from_numpy(tensor[2]/(tensor[2].max()+1e-10)).repeat(3,1,1)
        x.extend([pil_to_tensor(pil_input.convert('RGB')), pil_label, pil_output])
    x = torch.stack(x, 0)
    x = vutils.make_grid(x, nrow=3, padding=5)
    x = (x.numpy()*255).astype(np.uint8)

    writer.add_image(exp_name + '_epoch_' + str(epoch+1), x)



def print_summary(exp_name,scores,train_record):
    mae, mse, mgape, loss = scores
    print( '='*100 )
    print( exp_name )
    print( '    '+ '-'*60 )
    print( '    [mae %.2f rmse %.2f mgape %.2f], [val loss %.4f]' % (mae, mse, mgape, loss) )        
    print( '    '+ '-'*60 )
    print( '[best] [model: %s] , [mae %.2f], [rmse %.2f], [mgape %.2f]' % (train_record['best_model_name'],\
                                                        train_record['best_mae'],\
                                                        train_record['best_mse'],\
                                                        train_record['best_mgape']) )
    print( '='*100)

def print_WE_summary(log_txt,epoch,scores,train_record,c_maes):
    mae, mse, mgape, loss = scores
    # pdb.set_trace()
    with open(log_txt, 'a') as f:
        f.write('='*15 + '+'*15 + '='*15 + '\n')
        f.write(str(epoch) + '\n\n')
        f.write('  [mae %.4f], [val loss %.4f]\n\n' % (mae, loss))
        f.write('    list: ' + str(np.transpose(c_maes.avg)) + '\n')


        f.write('='*15 + '+'*15 + '='*15 + '\n\n')

    print( '='*50 )
    print( '    '+ '-'*20 )
    print( '    [mae %.2f rmse %.2f mgape %.2f], [val loss %.4f]' % (mae, mse, mgape, loss) )        
    print( '    '+ '-'*20 )
    print( '[best] [model: %s] , [mae %.2f], [rmse %.2f], [mgape %.2f]' % (train_record['best_model_name'],\
                                                        train_record['best_mae'],\
                                                        train_record['best_mse'],\
                                                        train_record['best_mgape']) )
    print( '='*50 )


def print_GCC_summary(log_txt,epoch, scores,train_record,c_maes,c_mses):
    mae, mse, mgape, loss = scores
    #c_mses['level'] = np.sqrt(c_mses['level'].avg)
    #c_mses['time'] = np.sqrt(c_mses['time'].avg)
    #c_mses['weather'] = np.sqrt(c_mses['weather'].avg)
    with open(log_txt, 'a') as f:
        f.write('='*15 + '+'*15 + '='*15 + '\n')
        f.write(str(epoch) + '\n\n')
        f.write('  [mae %.4f rmse %.4f mgape %.4f], [val loss %.4f]\n\n' % (mae, mse, mgape, loss))
        #f.write('  [level: mae %.4f mse %.4f]\n' % (np.average(c_maes['level'].avg), np.average(c_mses['level'])))
        #f.write('    list: ' + str(np.transpose(c_maes['level'].avg)) + '\n')
        #f.write('    list: ' + str(np.transpose(c_mses['level'])) + '\n\n')

        #f.write('  [time: mae %.4f mse %.4f]\n' % (np.average(c_maes['time'].avg), np.average(c_mses['time'])))
        #f.write('    list: ' + str(np.transpose(c_maes['time'].avg)) + '\n')
        #f.write('    list: ' + str(np.transpose(c_mses['time'])) + '\n\n')

        #f.write('  [weather: mae %.4f mse %.4f]\n' % (np.average(c_maes['weather'].avg), np.average(c_mses['weather'])))
        #f.write('    list: ' + str(np.transpose(c_maes['weather'].avg)) + '\n')
        #f.write('    list: ' + str(np.transpose(c_mses['weather']))+ '\n\n')

        #f.write('='*15 + '+'*15 + '='*15 + '\n\n')

    print( '='*50 )
    print( '    '+ '-'*20 )
    print( '    [mae %.2f rmse %.2f mgape %.2f], [val loss %.4f]' % (mae, mse, mgape, loss) )
    print( '    '+ '-'*20 )
    print( '[best] [model: %s] , [mae %.2f], [mse %.2f], [mgape %.2f]' % (train_record['best_model_name'],\
                                                        train_record['best_mae'],\
                                                        train_record['best_mse'],\
                                                        train_record['best_mgape']) )
    print( '='*50 )   


def update_model(net,optimizer,scheduler,epoch,i_tb,exp_path,exp_name,scores,train_record,log_file=None, best_metric='best_mae'):

    mae, mse, mgape, loss = scores

    snapshot_name = 'all_ep_%d_mae_%.1f_rmse_%.1f_mgape_%.1f' % (epoch + 1, mae, mse, mgape)
    
    metric = mae
    #TODO a adapter pour changer de comparaison de metrique
    if metric < train_record[best_metric]:
        best_state = {'train_record':train_record, 'net':net.state_dict(), 'optimizer':optimizer.state_dict(),\
                'scheduler':scheduler.state_dict(), 'epoch': epoch, 'i_tb':i_tb, 'exp_path':exp_path, \
                'exp_name':exp_name}
        train_record['best_model_name'] = snapshot_name
        train_record['best_mae'] = mae
        train_record['best_mse'] = mse 
        train_record['best_mgape'] = mgape
        train_record['best_loss'] = loss   
        if log_file is not None:
            logger_txt(log_file,epoch,scores)
        #to_saved_weight = net.state_dict()
        torch.save(best_state, os.path.join(exp_path, exp_name, 'best_state.pth'))
        
    latest_state = {'train_record':train_record, 'net':net.state_dict(), 'optimizer':optimizer.state_dict(),\
                    'scheduler':scheduler.state_dict(), 'epoch': epoch, 'i_tb':i_tb, 'exp_path':exp_path, \
                    'exp_name':exp_name}

    torch.save(latest_state,os.path.join(exp_path, exp_name, 'latest_state.pth'))

    return train_record


def copy_cur_env(work_dir, dst_dir, exception):
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    for filename in os.listdir(work_dir):

        file = os.path.join(work_dir, filename)

        valid = True
        for exc in exception:
            if exc == filename:
                valid = False
                break
        if not valid:
            continue

        dst_file = os.path.join(dst_dir, filename)

        if os.path.isdir(file):
            shutil.copytree(file, dst_file)
        elif os.path.isfile(file):
            shutil.copyfile(file, dst_file)




class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.cur_val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, cur_val):
        self.cur_val = cur_val
        self.sum += cur_val
        self.count += 1
        self.avg = self.sum / self.count

class AverageCategoryMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self,num_class):        
        self.num_class = num_class
        self.reset()

    def reset(self):
        self.cur_val = np.zeros(self.num_class)
        self.avg = np.zeros(self.num_class)
        self.sum = np.zeros(self.num_class)
        self.count = np.zeros(self.num_class)

    def update(self, cur_val, class_id):
        self.cur_val[class_id] = cur_val
        self.sum[class_id] += cur_val
        self.count[class_id] += 1
        self.avg[class_id] = self.sum[class_id] / self.count[class_id]


class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def get_grid_metrics(prediction_map, ground_truth_map, metric_grid, debug=False):
    
    if debug:
        print('metric_grid:',metric_grid)
        print("prediction_map(sum):",prediction_map.sum())     
        print('prediction_map.shape',prediction_map.shape)
        print('prediction_map',prediction_map)
        print("ground_truth_map(sum):",ground_truth_map.sum())     
        print('ground_truth_map.shape:',ground_truth_map.shape)
        print('ground_truth_map:',ground_truth_map)
    
    matrix_prediction_map = np.zeros(metric_grid)   
    
    pm_width = prediction_map.shape[1]
    pm_height = prediction_map.shape[0]
    if debug:
        print('pm_width:',pm_width)
        print('pm_height:',pm_height)
    
    n_w = int(math.ceil(pm_width/metric_grid[0]))
    n_h = int(math.ceil(pm_height/metric_grid[1]))
    for iw in range(metric_grid[0]):

        x_start = iw*n_w
        x_stop = (iw+1)*n_w
        if x_stop>pm_width:
            x_stop=pm_width

        for ih in range(metric_grid[1]):

            y_start = ih*n_h
            y_stop = (ih+1)*n_h
            if y_stop>pm_height:
                y_stop=pm_height

            sub_prediction_map = prediction_map[y_start:y_stop,x_start:x_stop]
            matrix_prediction_map[iw,ih] = sub_prediction_map.sum()
            if debug:
                print('iw:',iw,'x_start:',x_start,'x_stop:',x_stop,'ih:',ih,'y_start:',y_start,'y_stop:',y_stop)
                print("sub_prediction_map(sum):",sub_prediction_map.sum())     
                

    matrix_ground_truth_map = np.zeros(metric_grid)   
    
    gtm_width = ground_truth_map.shape[1]
    gtm_height = ground_truth_map.shape[0]
    if debug:
        print('gtm_width:',gtm_width)
        print('gtm_height:',gtm_height)
    
    n_w = int(math.ceil(gtm_width/metric_grid[0]))
    n_h = int(math.ceil(gtm_height/metric_grid[1]))
    for iw in range(metric_grid[0]):

        x_start = iw*n_w
        x_stop = (iw+1)*n_w
        if x_stop>gtm_width:
            x_stop=gtm_width

        for ih in range(metric_grid[1]):

            y_start = ih*n_h
            y_stop = (ih+1)*n_h
            if y_stop>gtm_height:
                y_stop=gtm_height

            sub_ground_truth_map = ground_truth_map[y_start:y_stop,x_start:x_stop]
            matrix_ground_truth_map[iw,ih] = sub_ground_truth_map.sum()
            if debug:
                print('iw:',iw,'x_start:',x_start,'x_stop:',x_stop,'ih:',ih,'y_start:',y_start,'y_stop:',y_stop)
                print("sub_ground_truth_map(sum):",sub_ground_truth_map.sum())              
     
    if debug:
        print('matrix_ground_truth_map:',matrix_ground_truth_map)
        print('matrix_prediction_map:',matrix_prediction_map)
        
    matrix_difference = matrix_ground_truth_map - matrix_prediction_map
    
    if debug:
        print('matrix_difference:',matrix_difference)
        
    matrix_final = matrix_difference.round()
    matrix_final = np.absolute(matrix_final)

    if debug:
        print('matrix_final:',matrix_final)
        
    gt_nb_person = ground_truth_map.sum()
    if gt_nb_person==0:
        gt_nb_person=1
    if debug:
        print('gt_nb_person:',gt_nb_person)
        
    #grid absolute percentage error
    gape = 100.*matrix_final.sum()/gt_nb_person
    if debug:
        print('gape:',gape)
        
    #grid cell absolute error
    gcae = (matrix_final.sum()/metric_grid[0]/metric_grid[1]).round()
    if debug:

        print('gcae:',gcae)
    return gape, gcae


def get_mean_and_std_by_channel(loader):
    
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for i, data in tqdm(enumerate(loader, 0)):

        img, gt_map = data
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(img, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(img ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    return list(mean.numpy()), list(std.numpy())


def get_mean_and_std_by_channel_2(loader):
    # Compute the mean and sd in an online fashion
    # Var[x] = E[X^2] - E^2[X]
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for i, data in tqdm(enumerate(loader, 0)):
        
        img, gt_map = data
        b, c, h, w = img.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2,
                                  dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        cnt += nb_pixels

    mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)
    return list(mean.numpy()), list(std.numpy())
