import sys

sys.path.append("../ia-foule-lab/")
from torch import optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

from misc.utils import *
from models.CC import CrowdCounter

from iafoule.metrics import get_metrics, get_metrics_with_points


class Trainer:
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

        self.train_record = {'best_mae': 1e20,
                             'best_mape': 1e20,
                             'best_mse': 1e20,
                             'best_mgape': 1e20,
                             'best_mgcae': 1e20,
                             'best_model_name': ''}
        self.train_record_golden = {'best_mae': 1e20,
                                    'best_mape': 1e20,
                                    'best_mse': 1e20,
                                    'best_mgape': 1e20,
                                    'best_mgcae': 1e20,
                                    'best_model_name': ''}

        self.timer = {'iter time': Timer(), 'train time': Timer(), 'val time': Timer(), 'val golden time': Timer()}

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

            print('train time: {:.2f}s'.format(self.timer['train time'].diff))
            print('=' * 80)

            # validation
            if epoch % self.cfg.VAL_FREQ == 0 or epoch > self.cfg.VAL_DENSE_START:
                self.timer['val time'].tic()
                best_model = False
                if self.data_mode in ['SHHA', 'SHHB', 'QNRF', 'UCF50', 'Multiple']:
                    best_model = self.validate_v1()
                elif self.data_mode == 'WE':
                    best_model = self.validate_v2()
                elif self.data_mode == 'GCC':
                    best_model = self.validate_v3()
                self.timer['val time'].toc(average=False)
                print('val time: {:.2f}s'.format(self.timer['val time'].diff))

                if best_model and self.cfg.INFER_GOLDEN_DATASET:
                    self.timer['val golden time'].tic()
                    self.validate_golden()
                    self.timer['val golden time'].toc(average=False)
                    print('val golden time: {:.2f}s'.format(self.timer['val golden time'].diff))

    def train(self):  # training for all datasets
        train_losses = AverageMeter()
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
                self.writer.add_scalar('train_loss_batch', loss.item(), self.i_tb)
                self.timer['iter time'].toc(average=False)
                print('[ep %d][it %d][loss %.4f][lc_loss %.4f][lr %.4f][%.2fs]' %
                      (self.epoch + 1, i + 1, loss.item(), lc_loss, self.optimizer.param_groups[0]['lr'] * 10000,
                       self.timer['iter time'].diff))
                print('        [cnt: gt: %.1f pred: %.2f]' % (
                    gt_map[0].sum().data / self.cfg.LOG_PARA, pred_map[0].sum().data / self.cfg.LOG_PARA))

            train_losses.update(loss)
        train_loss = train_losses.avg
        self.writer.add_scalar('train_loss', train_loss, self.epoch + 1)

    def validate_v1(self):
        """
        validate_V1 for SHHA, SHHB, UCF-QNRF, UCF50
        """
        self.net.eval()

        losses = AverageMeter()
        maes = AverageMeter()
        mapes = AverageMeter()
        mses = AverageMeter()
        mgapes = AverageMeter()
        mgcaes = AverageMeter()

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
                    pred_cnt = np.sum(pred_map[i_img]) / self.cfg.LOG_PARA
                    gt_count = np.sum(gt_map[i_img]) / self.cfg.LOG_PARA

                    losses.update(self.net.loss.item())
                    
                    metric_grids = [(4, 4)]
                    metrics = get_metrics(pred_map[i_img].squeeze() / self.cfg.LOG_PARA,
                                        gt_map[i_img] / self.cfg.LOG_PARA, 
                                        metric_grids=metric_grids)
                        
                    maes.update(metrics['absolute_error'])
                    mapes.update(metrics['absolute_percentage_error'])
                    mses.update(metrics['squared_error'])
                    mgapes.update(metrics['grid4x4_absolute_percentage_error'])
                    mgcaes.update(metrics['grid4x4_cell_absolute_error'])
                    
                if vi == -1:
                    
                    vis_results(self.exp_name,
                                self.epoch,
                                self.writer,
                                self.restore_transform,
                                img,
                                pred_map,
                                gt_map)

        mae = maes.avg
        mape = mapes.avg
        mse = np.sqrt(mses.avg)
        loss = losses.avg
        mgape = mgapes.avg
        mgcae = mgcaes.avg

        self.writer.add_scalar('val_loss', loss, self.epoch + 1)
        self.writer.add_scalar('mae', mae, self.epoch + 1)
        self.writer.add_scalar('mape', mape, self.epoch + 1)
        self.writer.add_scalar('rmse', mse, self.epoch + 1)
        self.writer.add_scalar('mgape', mgape, self.epoch + 1)
        self.writer.add_scalar('mgcae', mgcae, self.epoch + 1)

        best_model = False
        best_metric = 'best_mae'
        if mae < self.train_record[best_metric]:
            self.train_record, best_model = update_model(self.net, self.optimizer, self.scheduler, self.epoch,
                                                         self.i_tb, self.exp_path,
                                                         self.exp_name,
                                                         [mae, mape, mse, mgape, mgcae, loss], self.train_record,
                                                         self.log_txt,
                                                         best_metric=best_metric)
            tr = self.train_record
            table_validation = f"""
### Table des métriques Validation
| **Best MAE** | **Best MAPE** | **Best RMSE** | **Best MGAPE** | **Best MGCAE** |
| ---- | ---- | ---- | ---- | ---- |
| {tr['best_mae']} | {tr['best_mape']} | {tr['best_mse']} | {tr['best_mgape']} | {tr['best_mgcae']} | 
"""
            self.writer.add_text("validation_table", table_validation, global_step=self.epoch + 1)

        print_summary(self.exp_name, [mae, mape, mse, mgape, mgcae, loss], self.train_record)

        return best_model

    def validate_v2(self):
        """
        validate_V2 for WE
        """
        self.net.eval()

        losses = AverageCategoryMeter(5)
        maes = AverageCategoryMeter(5)
        mapes = AverageCategoryMeter(5)
        mses = AverageCategoryMeter(5)
        mgapes = AverageCategoryMeter(5)
        mgcaes = AverageCategoryMeter(5)
        
        #roi_mask = []
        from datasets.WE.setting import cfg_data
        #from scipy import io as sio
        #for val_folder in cfg_data.VAL_FOLDER:
        #    roi_mask.append(sio.loadmat(os.path.join(cfg_data.DATA_PATH, 'test', val_folder + '_roi.mat'))['BW'])

        for i_sub, i_loader in enumerate(self.val_loader, 0):

            # mask = roi_mask[i_sub]
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
                        pred_cnt = np.sum(pred_map[i_img]) / self.cfg.LOG_PARA
                        gt_count = np.sum(gt_map[i_img]) / self.cfg.LOG_PARA

                        #losses.update(self.net.loss.item(), i_sub)
                        #maes.update(abs(gt_count - pred_cnt), i_sub)
                        
                        losses.update(self.net.loss.item(), i_sub)
                    
                        metric_grids = [(4, 4)]
                        metrics = get_metrics(pred_map[i_img].squeeze() / self.cfg.LOG_PARA,
                                            gt_map[i_img] / self.cfg.LOG_PARA, 
                                            metric_grids=metric_grids)
                        
                        maes.update(metrics['absolute_error'], i_sub)
                        mapes.update(metrics['absolute_percentage_error'], i_sub)
                        mses.update(metrics['squared_error'], i_sub)
                        mgapes.update(metrics['grid4x4_absolute_percentage_error'], i_sub)
                        mgcaes.update(metrics['grid4x4_cell_absolute_error'], i_sub)
                        
                    if vi == -1:
                        vis_results(self.exp_name,
                                    self.epoch,
                                    self.writer,
                                    self.restore_transform,
                                    img, pred_map,
                                    gt_map)

        #mae = np.average(maes.avg)
        #loss = np.average(losses.avg)

        mae = np.average(maes.avg)
        mape = np.average(mapes.avg)
        mse = np.sqrt(np.average(mses.avg))
        loss = np.average(losses.avg)
        mgape = np.average(mgapes.avg)
        mgcae = np.average(mgcaes.avg)

        #self.writer.add_scalar('val_loss', loss, self.epoch + 1)
        #self.writer.add_scalar('mae', mae, self.epoch + 1)
        #self.writer.add_scalar('mae_s1', maes.avg[0], self.epoch + 1)
        #self.writer.add_scalar('mae_s2', maes.avg[1], self.epoch + 1)
        #self.writer.add_scalar('mae_s3', maes.avg[2], self.epoch + 1)
        #self.writer.add_scalar('mae_s4', maes.avg[3], self.epoch + 1)
        #self.writer.add_scalar('mae_s5', maes.avg[4], self.epoch + 1)

        self.writer.add_scalar('val_loss', loss, self.epoch + 1)
        self.writer.add_scalar('mae', mae, self.epoch + 1)
        self.writer.add_scalar('mape', mape, self.epoch + 1)
        self.writer.add_scalar('rmse', mse, self.epoch + 1)
        self.writer.add_scalar('mgape', mgape, self.epoch + 1)
        self.writer.add_scalar('mgcae', mgcae, self.epoch + 1)
        
        #self.train_record = update_model(self.net, self.optimizer, self.scheduler, self.epoch, self.i_tb, self.exp_path,
        #                                 self.exp_name,[mae, 0, 0, 0, 0, loss], self.train_record, self.log_txt)

        best_model = False
        best_metric = 'best_mae'
        if mae < self.train_record[best_metric]:
            self.train_record, best_model = update_model(self.net,
                                                         self.optimizer,
                                                         self.scheduler,
                                                         self.epoch,
                                                         self.i_tb,
                                                         self.exp_path,
                                                         self.exp_name,
                                                         [mae, mape, mse, mgape, mgcae, loss],
                                                         self.train_record,
                                                         self.log_txt,
                                                         best_metric=best_metric)
            tr = self.train_record
            table_validation = f"""
### Table des métriques Validation
| **Best MAE** | **Best MAPE** | **Best RMSE** | **Best MGAPE** | **Best MGCAE** |
| ---- | ---- | ---- | ---- | ---- |
| {tr['best_mae']} | {tr['best_mape']} | {tr['best_mse']} | {tr['best_mgape']} | {tr['best_mgcae']} | 
"""
            self.writer.add_text("validation_table", table_validation, global_step=self.epoch + 1)

        #print_WE_summary(self.log_txt, self.epoch, [mae, 0, 0, 0, 0, loss], self.train_record, maes)
        print_summary(self.exp_name, [mae, mape, mse, mgape, mgcae, loss], self.train_record)

        return best_model

    def validate_v3(self):
        """
        validate_V3 for GCC
        """
        self.net.eval()

        losses = AverageMeter()
        maes = AverageMeter()
        mses = AverageMeter()

        c_maes = {'level': AverageCategoryMeter(9), 'time': AverageCategoryMeter(8), 'weather': AverageCategoryMeter(7)}
        c_mses = {'level': AverageCategoryMeter(9), 'time': AverageCategoryMeter(8), 'weather': AverageCategoryMeter(7)}

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
                    pred_cnt = np.sum(pred_map[i_img]) / self.cfg.LOG_PARA
                    gt_count = np.sum(gt_map[i_img]) / self.cfg.LOG_PARA

                    s_mae = abs(gt_count - pred_cnt)
                    s_mse = (gt_count - pred_cnt) * (gt_count - pred_cnt)

                    losses.update(self.net.loss.item())
                    maes.update(s_mae)
                    mses.update(s_mse)
                    # TODO : Debug
                    # attributes_pt = np.squeeze(attributes_pt)
                    # c_maes['level'].update(s_mae,attributes_pt[i_img][0])
                    # c_mses['level'].update(s_mse,attributes_pt[i_img][0])
                    # c_maes['time'].update(s_mae,attributes_pt[i_img][1]/3)
                    # c_mses['time'].update(s_mse,attributes_pt[i_img][1]/3)
                    # c_maes['weather'].update(s_mae,attributes_pt[i_img][2])
                    # c_mses['weather'].update(s_mse,attributes_pt[i_img][2])

                if vi == -1:  # set to 0 if u want show image
                    vis_results(self.exp_name, self.epoch, self.writer, self.restore_transform, img, pred_map, gt_map)

        loss = losses.avg
        mae = maes.avg
        mse = np.sqrt(mses.avg)

        self.writer.add_scalar('val_loss', loss, self.epoch + 1)
        self.writer.add_scalar('mae', mae, self.epoch + 1)
        self.writer.add_scalar('rmse', mse, self.epoch + 1)

        self.train_record_golden = update_model(self.net, self.optimizer, self.scheduler, self.epoch, self.i_tb,
                                                self.exp_path, self.exp_name,
                                                [mae, 0, mse, 0, 0, loss], self.train_record_golden, self.log_txt)

        print_GCC_summary(self.log_txt, self.epoch, [mae, 0, mse, 0, 0, loss], self.train_record_golden, c_maes, c_mses)

        return False

    def validate_golden(self):
        """
        validate_golden Validate for golden dataset
        """
        from datasets.GD.loading_data import loading_data

        self.net.eval()

        maes = AverageMeter()
        mapes = AverageMeter()
        mses = AverageMeter()
        mgapes = AverageMeter()
        mgcaes = AverageMeter()

        golden_val_loader = loading_data()

        for vi, data in enumerate(golden_val_loader, 0):
            img, gt_count, gt_points = data

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
                    pred_cnt = np.sum(pred_map[i_img]) / self.cfg.LOG_PARA
                    
                    width = img.shape[3]
                    height = img.shape[2]
                    ground_truth = [(point['x'].item(), point['y'].item()) for point in gt_points]
                    metric_grids = [(4, 4)]                   
                    metrics = get_metrics_with_points(pred_map[i_img].squeeze() / self.cfg.LOG_PARA,
                                          ground_truth, metric_grids=metric_grids)

                    maes.update(metrics['absolute_error'])
                    mapes.update(metrics['absolute_percentage_error'])
                    mses.update(metrics['squared_error'])
                    mgapes.update(metrics['grid4x4_absolute_percentage_error'])
                    mgcaes.update(metrics['grid4x4_cell_absolute_error'])
                    
                    #maes.update(abs(gt_count - pred_cnt))
                    #if gt_count == 0.:
                    #    ape = 100. * abs(gt_count - pred_cnt)
                    #else:
                    #    ape = 100. * abs(gt_count - pred_cnt) / gt_count
                    #mapes.update(ape)
                    #mses.update((gt_count - pred_cnt) * (gt_count - pred_cnt))

                    #gape, gcae = get_grid_metrics_with_points(width, height,
                    #                                          pred_map[i_img].squeeze() / self.cfg.LOG_PARA,
                    #                                          ground_truth,  # gt_count[i_img],
                    #                                          metric_grids[0],
                    #                                          debug=False)
                    #mgapes.update(gape)
                    #mgcaes.update(gcae)
                    #print('AE:', metrics['absolute_error'], abs(pred_cnt - gt_count))
                    #print('APE:', metrics['absolute_percentage_error'], ape)
                    #print('SE:', metrics['squared_error'], ((gt_count - pred_cnt) * (gt_count - pred_cnt)))
                    #print('GAPE:', metrics['grid4x4_absolute_percentage_error'], gape)
                    #print('GCAE:', metrics['grid4x4_cell_absolute_error'], gcae)
                    
        mae = maes.avg
        mape = mapes.avg
        mse = np.sqrt(mses.avg)
        loss = 0
        mgape = mgapes.avg
        mgcae = mgcaes.avg

        self.writer.add_scalar('mae_golden', mae, self.epoch + 1)
        self.writer.add_scalar('mape_golden', mape, self.epoch + 1)
        self.writer.add_scalar('rmse_golden', mse, self.epoch + 1)
        self.writer.add_scalar('mgape_golden', mgape, self.epoch + 1)
        self.writer.add_scalar('mgcae_golden', mgcae, self.epoch + 1)

        self.train_record_golden['best_mae'] = mae.item()
        self.train_record_golden['best_mape'] = mape.item()
        self.train_record_golden['best_mse'] = mse.item()
        self.train_record_golden['best_mgape'] = mgape.item()
        self.train_record_golden['best_mgcae'] = mgcae.item()

        tr = self.train_record_golden
        table_golden = f"""
### Table des métriques Golden
| **Best MAE** | **Best MAPE** | **Best RMSE** | **Best MGAPE** | **Best MGCAE** |
| ---- | ---- | ---- | ---- | ---- |
| {tr['best_mae']} | {tr['best_mape']} | {tr['best_mse']} | {tr['best_mgape']} | {tr['best_mgcae']} | 
"""

        self.writer.add_text("validation_golden", table_golden, global_step=self.epoch + 1)

        print_summary(self.exp_name + "-Golden", [mae, mape, mse, mgape, mgcae, loss], self.train_record_golden)
