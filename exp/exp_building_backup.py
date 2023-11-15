import csv

from tqdm import tqdm

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import DLinear, SLinear, DCLinear
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')


class Building_Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Building_Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'DLinear': DLinear,
            'SLinear': SLinear,
            'CNNLinear': CNNLinear,
            'NSTLCNNLinear': NSTLCNNLinear
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            # 使用 tqdm 创建一个进度条，并包装 vali_loader
            for i, (seasonal, trend, residual, target, features, time_stamp, pred_target, pred_features,
                    pred_timestamp) in enumerate(
                    vali_loader):

                seasonal = seasonal.float().to(self.device)
                trend = trend.float().to(self.device)
                residual = residual.float().to(self.device)
                target = target.float().to(self.device)
                features = features.float().to(self.device)
                time_stamp = time_stamp.float().to(self.device)
                pred_target = pred_target.float().to(self.device)
                pred_features = pred_features.float().to(self.device)
                pred_timestamp = pred_timestamp.float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(seasonal, trend, residual, target, features, time_stamp, pred_features,
                                             pred_timestamp)
                        loss = criterion(outputs, pred_target)
                else:
                    outputs = self.model(seasonal, trend, residual, target, features, time_stamp, pred_target, pred_features, pred_timestamp)
                    loss = criterion(outputs, pred_target)

                pred_target = pred_target.to(self.device)

                pred = outputs.detach().cpu()
                true = pred_target.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        if not self.args.train_only:
            vali_data, vali_loader = self._get_data(flag='val')
            test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            # 使用 tqdm 创建一个进度条，并包装 train_loader
            loop_loader = tqdm((train_loader), total=len(train_loader))
            for i, (true_target, true_features, true_time_mask, pred_target) in loop_loader:
                # 记录开始时间
                # start_time = time.time()
                iter_count += 1
                model_optim.zero_grad()
                seasonal = seasonal.float().to(self.device)
                trend = trend.float().to(self.device)
                residual = residual.float().to(self.device)
                target = target.float().to(self.device)
                features = features.float().to(self.device)
                time_stamp = time_stamp.float().to(self.device)
                pred_target = pred_target.float().to(self.device)
                pred_features = pred_features.float().to(self.device)
                pred_timestamp = pred_timestamp.float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(seasonal, trend, residual, target, features, time_stamp, pred_features,
                                             pred_timestamp)
                        loss = criterion(outputs, pred_target)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(seasonal, trend, residual, target, features, time_stamp, pred_features,
                                         pred_timestamp,pred_timestamp)
                    loss = criterion(outputs, pred_target)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            if not self.args.train_only:
                vali_loss = self.vali(vali_data, vali_loader, criterion)
                test_loss = self.vali(test_data, test_loader, criterion)

                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                early_stopping(vali_loss, self.model, path)
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                    epoch + 1, train_steps, train_loss))
                early_stopping(train_loss, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, args, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            # 使用 tqdm 创建一个进度条，并包装 vali_loader
            for i, (seasonal, trend, residual, target, features, time_stamp, pred_target, pred_features,
                    pred_timestamp) in enumerate(
                    test_loader):

                seasonal = seasonal.float().to(self.device)
                trend = trend.float().to(self.device)
                residual = residual.float().to(self.device)
                target = target.float().to(self.device)
                features = features.float().to(self.device)
                time_stamp = time_stamp.float().to(self.device)
                pred_target = pred_target.float().to(self.device)
                pred_features = pred_features.float().to(self.device)
                pred_timestamp = pred_timestamp.float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(seasonal, trend, residual, target, features, time_stamp, pred_features,
                                             pred_timestamp)
                else:
                    outputs = self.model(seasonal, trend, residual, target, features, time_stamp, pred_target, pred_features, pred_timestamp)

                pred_target = pred_target.to(self.device)
                outputs = outputs.detach().cpu().numpy()
                pred_target = pred_target.detach().cpu().numpy()
                #
                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = pred_target  # batch_y.detach().cpu().numpy()  # .squeeze()
                #
                preds.append(pred)
                trues.append(true)
                inputx.append(target.detach().cpu().numpy())
                if i % 20 == 0:
                    input = target.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :], true[0, :]), axis=0)
                    pd = np.concatenate((input[0, :], pred[0, :]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
        #
        if self.args.test_flop:
            test_params_flop((target.shape[1], target.shape[2]))
            exit()

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, r2 = metric(preds, trues)
        print('mae:{}, mse:{}, rmse:{}, mape:{}, mspe:{}, rse:{}, r2:{}'.format(mae, mse, rmse, mape, mspe, rse, r2))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mae:{}, mse:{}, rmse:{}, mape:{}, mspe:{}, rse:{}, r2:{}'.format(mae, mse, rmse, mape, mspe, rse, r2))
        f.write('\n')
        f.write('\n')
        f.close()

        res_csv = [args.model, args.seq_len, args.pred_len, mae, mse, rmse, mape, mspe, rse, r2]
        # CSV 文件路径
        csv_file_path = "results.csv"

        # 打开 CSV 文件以追加新的行
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(res_csv)

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(
                    batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = np.concatenate(preds, axis=0)
        if (pred_data.scale):
            preds = pred_data.inverse_transform(preds)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)
        pd.DataFrame(np.append(np.transpose([pred_data.future_dates]), preds[0], axis=1),
                     columns=pred_data.cols).to_csv(folder_path + 'real_prediction.csv', index=False)

        return
