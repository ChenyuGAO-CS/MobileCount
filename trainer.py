import sys
import numpy as np

import torch
from torch import optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

from models.CC import CrowdCounter
#from config import cfg
from misc.utils import *
import pdb


class Trainer():
    def __init__(self, dataloader, cfg, pwd):

        self.cfg = cfg

        self.data_mode = cfg.DATASET
        self.exp_name = cfg.EXP_NAME
        self.exp_path = cfg.EXP_PATH
        
        self.pwd = pwd

        self.net_name = cfg.NET
        self.net = CrowdCounter(cfg.GPU_ID, self.net_name, cfg=cfg)
        if torch.cuda.is_available():    
            self.net.cuda()
            
        self.optimizer = optim.Adam(self.net.CCN.parameters(), lr=cfg.LR, weight_decay=1e-4)
        # self.optimizer = optim.SGD(self.net.parameters(), cfg.LR, momentum=0.95,weight_decay=5e-4)
        self.scheduler = StepLR(self.optimizer, step_size=cfg.NUM_EPOCH_LR_DECAY, gamma=cfg.LR_DECAY)          

        self.train_record = {'best_mae': 1e20, 'best_mse':1e20, 'best_mgape':1e20, 'best_model_name': ''}
        self.train_record_golden = {'best_mae': 1e20, 'best_mse':1e20, 'best_mgape':1e20, 'best_model_name': ''}

        self.timer = {'iter time' : Timer(),'train time' : Timer(),'val time' : Timer()} 
        
        self.epoch = 0
        self.i_tb = 0
        
        if cfg.PRE_GCC:
            self.net.load_state_dict(torch.load(cfg.PRE_GCC_MODEL))

        self.train_loader, self.val_loader, self.restore_transform = dataloader()

        if cfg.RESUME:
            latest_state = torch.load(cfg.RESUME_PATH)
            self.net.load_state_dict(latest_state['net'])
            self.optimizer.load_state_dict(latest_state['optimizer'])
            self.scheduler.load_state_dict(latest_state['scheduler'])
            self.epoch = latest_state['epoch'] + 1
            self.i_tb = latest_state['i_tb']
            self.train_record = latest_state['train_record']
            self.exp_path = latest_state['exp_path']
            self.exp_name = latest_state['exp_name']

        self.writer, self.log_txt = logger(self.exp_path, 
                                           self.exp_name, 
                                           self.pwd, ['exp', '.git', '.idea'], 
                                           resume=cfg.RESUME)


    def forward(self):

        for epoch in range(self.epoch, self.cfg.MAX_EPOCH):
            self.epoch = epoch
            if epoch > self.cfg.LR_DECAY_START:
                self.scheduler.step()
                
            # training    
            self.timer['train time'].tic()
            self.train()
            self.timer['train time'].toc(average=False)

            print( 'train time: {:.2f}s'.format(self.timer['train time'].diff) )
            print( '='*80 )

            # validation
            if epoch%self.cfg.VAL_FREQ==0 or epoch>self.cfg.VAL_DENSE_START:
                self.timer['val time'].tic()
                if self.data_mode in ['SHHA', 'SHHB', 'QNRF', 'UCF50']:
                    self.validate_V1()
                elif self.data_mode == 'WE':
                    self.validate_V2()
                elif self.data_mode == 'GCC':
                    self.validate_V3()
                self.timer['val time'].toc(average=False)
                print( 'val time: {:.2f}s'.format(self.timer['val time'].diff) )
                
                if self.cfg.INFER_GOLDEN_DATASET:
                    self.validate_GD()

    def train(self): # training for all datasets
        self.net.train()
        for i, data in enumerate(self.train_loader, 0):
            self.timer['iter time'].tic()
            img, gt_map = data
            img = Variable(img)
            gt_map = Variable(gt_map)
            
            if torch.cuda.is_available():
                img = img.cuda()
                gt_map = gt_map.cuda()
            
            self.optimizer.zero_grad()
            pred_map = self.net(img, gt_map)
            loss = self.net.loss
            if isinstance(self.net.lc_loss, int):
                lc_loss = self.net.lc_loss
            else:
                lc_loss = self.net.lc_loss.item()
            loss.backward()
            self.optimizer.step()

            if (i + 1) % self.cfg.PRINT_FREQ == 0:            
                self.i_tb += 1
                self.writer.add_scalar('train_loss', loss.item(), self.i_tb)
                self.timer['iter time'].toc(average=False)
                print( '[ep %d][it %d][loss %.4f][lc_loss %.4f][lr %.4f][%.2fs]' % \
                        (self.epoch + 1, i + 1, loss.item(), lc_loss, self.optimizer.param_groups[0]['lr']*10000, self.timer['iter time'].diff) )
                print( '        [cnt: gt: %.1f pred: %.2f]' % (gt_map[0].sum().data/self.cfg.LOG_PARA, pred_map[0].sum().data/self.cfg.LOG_PARA) )           


    def validate_V1(self, metric_grid=(4,4)):# validate_V1 for SHHA, SHHB, UCF-QNRF, UCF50

        self.net.eval()
        
        losses = AverageMeter()
        maes = AverageMeter()
        mses = AverageMeter()
        mgapes = AverageMeter()

        for vi, data in enumerate(self.val_loader, 0):
            img, gt_map = data

            with torch.no_grad():
                img = Variable(img)
                gt_map = Variable(gt_map)

                if torch.cuda.is_available():
                    img = img.cuda()
                    gt_map = gt_map.cuda()
                pred_map = self.net.forward(img, gt_map)
                if torch.cuda.is_available():
                    pred_map = pred_map.data.cpu().numpy()
                    gt_map = gt_map.data.cpu().numpy()
                else:
                    pred_map = pred_map.data.numpy()
                    gt_map = gt_map.data.numpy()

                for i_img in range(pred_map.shape[0]):
                
                    pred_cnt = np.sum(pred_map[i_img])/self.cfg.LOG_PARA
                    gt_count = np.sum(gt_map[i_img])/self.cfg.LOG_PARA

                    losses.update(self.net.loss.item())
                    maes.update(abs(gt_count-pred_cnt))
                    mses.update((gt_count-pred_cnt)*(gt_count-pred_cnt))
                    
                    gape, gcae = get_grid_metrics(pred_map[i_img].squeeze()/self.cfg.LOG_PARA, 
                                                gt_map[i_img]/self.cfg.LOG_PARA, 
                                                metric_grid, 
                                                debug=False)
                    mgapes.update(gape)
                    
                if vi==-1: # -1
                    vis_results(self.exp_name, self.epoch, self.writer, self.restore_transform, img, pred_map, gt_map)
            
        mae = maes.avg
        mse = np.sqrt(mses.avg)
        loss = losses.avg
        mgape = mgapes.avg
        
        self.writer.add_scalar('val_loss', loss, self.epoch + 1)
        self.writer.add_scalar('mae', mae, self.epoch + 1)
        self.writer.add_scalar('rmse', mse, self.epoch + 1)
        self.writer.add_scalar('mgape', mgape, self.epoch + 1)
        
        self.train_record = update_model(self.net,self.optimizer,self.scheduler,self.epoch,self.i_tb,self.exp_path,self.exp_name, \
            [mae, mse, mgape, loss],self.train_record,self.log_txt)
        
        self.TABLE_VALID = f"""
### Table des métriques Validation

| **Best MAE** | **Best RMSE** | **Best MGAPE** |
| ---- | ---- | ---- |
| {self.train_record['best_mae']} | {self.train_record['best_mse']} | {self.train_record['best_mgape']} | 

"""
        
        self.writer.add_text("validation_table", self.TABLE_VALID, global_step=self.epoch + 1)
        print_summary(self.exp_name,[mae, mse, mgape, loss],self.train_record)


    def validate_V2(self):# validate_V2 for WE

        self.net.eval()

        losses = AverageCategoryMeter(5)
        maes = AverageCategoryMeter(5)

        roi_mask = []
        from datasets.WE.setting import cfg_data 
        from scipy import io as sio
        for val_folder in cfg_data.VAL_FOLDER:

            roi_mask.append(sio.loadmat(os.path.join(cfg_data.DATA_PATH,'test',val_folder + '_roi.mat'))['BW'])
        
        for i_sub,i_loader in enumerate(self.val_loader,0):

            mask = roi_mask[i_sub]
            for vi, data in enumerate(i_loader, 0):
                img, gt_map = data

                with torch.no_grad():
                    img = Variable(img)
                    gt_map = Variable(gt_map)

                    if torch.cuda.is_available():
                        img = img.cuda()
                        gt_map = gt_map.cuda()
                        
                    pred_map = self.net.forward(img, gt_map)
                    
                    if torch.cuda.is_available():
                        pred_map = pred_map.data.cpu().numpy()
                        gt_map = gt_map.data.cpu().numpy()
                    else:
                        pred_map = pred_map.data.numpy()
                        gt_map = gt_map.data.numpy()

                    for i_img in range(pred_map.shape[0]):
                    
                        pred_cnt = np.sum(pred_map[i_img])/self.cfg_data.LOG_PARA
                        gt_count = np.sum(gt_map[i_img])/self.cfg_data.LOG_PARA

                        losses.update(self.net.loss.item(),i_sub)
                        maes.update(abs(gt_count-pred_cnt),i_sub)
                    if vi==0:
                        vis_results(self.exp_name, self.epoch, self.writer, self.restore_transform, img, pred_map, gt_map)
            
        mae = np.average(maes.avg)
        loss = np.average(losses.avg)

        self.writer.add_scalar('val_loss', loss, self.epoch + 1)
        self.writer.add_scalar('mae', mae, self.epoch + 1)
        self.writer.add_scalar('mae_s1', maes.avg[0], self.epoch + 1)
        self.writer.add_scalar('mae_s2', maes.avg[1], self.epoch + 1)
        self.writer.add_scalar('mae_s3', maes.avg[2], self.epoch + 1)
        self.writer.add_scalar('mae_s4', maes.avg[3], self.epoch + 1)
        self.writer.add_scalar('mae_s5', maes.avg[4], self.epoch + 1)

        self.train_record = update_model(self.net,self.optimizer,self.scheduler,self.epoch,self.i_tb,self.exp_path,self.exp_name, \
            [mae, 0, 0, loss],self.train_record,self.log_txt)
        print_WE_summary(self.log_txt,self.epoch,[mae, 0, 0, loss],self.train_record,maes)





    def validate_V3(self):# validate_V3 for GCC

        self.net.eval()
        
        losses = AverageMeter()
        maes = AverageMeter()
        mses = AverageMeter()

        c_maes = {'level':AverageCategoryMeter(9), 'time':AverageCategoryMeter(8),'weather':AverageCategoryMeter(7)}
        c_mses = {'level':AverageCategoryMeter(9), 'time':AverageCategoryMeter(8),'weather':AverageCategoryMeter(7)}


        for vi, data in enumerate(self.val_loader, start=0):
            img, gt_map, attributes_pt = data

            with torch.no_grad():
                img = Variable(img)
                gt_map = Variable(gt_map)
                
                if torch.cuda.is_available():
                    img = img.cuda()
                    gt_map = gt_map.cuda()
                pred_map = self.net.forward(img, gt_map)
                
                if torch.cuda.is_available():
                    pred_map = pred_map.data.cpu().numpy()
                    gt_map = gt_map.data.cpu().numpy()
                else:
                    pred_map = pred_map.data.numpy()
                    gt_map = gt_map.data.numpy()

                for i_img in range(pred_map.shape[0]):
                
                    pred_cnt = np.sum(pred_map[i_img]) /self.cfg_data.LOG_PARA
                    gt_count = np.sum(gt_map[i_img]) /self.cfg_data.LOG_PARA

                    s_mae = abs(gt_count-pred_cnt)
                    s_mse = (gt_count-pred_cnt)*(gt_count-pred_cnt)

                    losses.update(self.net.loss.item())
                    maes.update(s_mae)
                    mses.update(s_mse)
                    # TODO : Debug
                    #attributes_pt = np.squeeze(attributes_pt)
                    #c_maes['level'].update(s_mae,attributes_pt[i_img][0])
                    #c_mses['level'].update(s_mse,attributes_pt[i_img][0])
                    #c_maes['time'].update(s_mae,attributes_pt[i_img][1]/3)
                    #c_mses['time'].update(s_mse,attributes_pt[i_img][1]/3)
                    #c_maes['weather'].update(s_mae,attributes_pt[i_img][2])
                    #c_mses['weather'].update(s_mse,attributes_pt[i_img][2])


                if vi == -1: # set to 0 if u want show image
                    vis_results(self.exp_name, self.epoch, self.writer, self.restore_transform, img, pred_map, gt_map)
            
        loss = losses.avg
        mae = maes.avg
        mse = np.sqrt(mses.avg)


        self.writer.add_scalar('val_loss', loss, self.epoch + 1)
        self.writer.add_scalar('mae', mae, self.epoch + 1)
        self.writer.add_scalar('rmse', mse, self.epoch + 1)

        self.train_record_golden = update_model(self.net,self.optimizer,self.scheduler,self.epoch,self.i_tb,self.exp_path,self.exp_name, \
            [mae, mse, 0, loss],self.train_record_golden,self.log_txt)


        print_GCC_summary(self.log_txt,self.epoch,[mae, mse, 0, loss],self.train_record_golden,c_maes,c_mses)

        
    def validate_GD(self):# validate_GD Validate for golden dataset
        from torch.utils.data import DataLoader
        from datasets.GD.loading_data import loading_data

        self.net.eval()
        
        losses = AverageMeter()
        maes = AverageMeter()
        mses = AverageMeter()
        
        golden_val_loader = loading_data()
        
        for vi, data in enumerate(golden_val_loader, 0):
            img, gt_count = data
            
            with torch.no_grad():
                img = Variable(img)
                
                if torch.cuda.is_available():
                    img = img.cuda()

                pred_map = self.net.forward(img)
                
                if torch.cuda.is_available():
                    pred_map = pred_map.data.cpu().numpy()
                else:
                    pred_map = pred_map.data.numpy()

                for i_img in range(pred_map.shape[0]):
                
                    pred_cnt = np.sum(pred_map[i_img])/self.cfg.LOG_PARA
                    #gt_count = np.sum(gt_map[i_img])/self.cfg_data.LOG_PARA

                    
                    #losses.update(self.net.loss.item())
                    maes.update(abs(gt_count-pred_cnt))
                    mses.update((gt_count-pred_cnt)*(gt_count-pred_cnt))
                #if vi==0:
                #    vis_results(self.exp_name, self.epoch, self.writer, self.restore_transform, img, pred_map, gt_map)
            
        mae = maes.avg
        mse = np.sqrt(mses.avg)
        loss = 0
        #loss = losses.avg

        #self.writer.add_scalar('val_loss_golden', loss, self.epoch + 1)
        self.writer.add_scalar('mae_golden', mae, self.epoch + 1)
        self.writer.add_scalar('mse_golden', mse, self.epoch + 1)
        
        if mae < self.train_record_golden['best_mae']:
            self.train_record_golden['best_mae'] = mae
            self.train_record_golden['best_mse'] = mse
            
            

        #self.train_record_golden = update_model(self.net,self.optimizer,self.scheduler,self.epoch,
        #self.i_tb,self.exp_path,self.exp_name,[mae, mse, 0, loss],self.train_record_golden, None)
        
        self.TABLE_GOLDEN = f"""
### Table des métriques Golden

| **Best MAE** | **Best RMSE** | **Best MGAPE** |
| ---- | ---- | ---- | 
| {self.train_record_golden['best_mae']} | {self.train_record_golden['best_mse']} | {self.train_record_golden['best_mgape']} | 

"""
        self.writer.add_text("validation_golden", self.TABLE_GOLDEN, global_step=self.epoch + 1)
        print_summary(self.exp_name,[mae, mse, 0, loss],self.train_record)
