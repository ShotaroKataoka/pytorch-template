import os
import shutil
from collections import OrderedDict
import glob

import torch

class Saver(object):

    def __init__(self, model_name, lr, epochs):
        self.model_name = model_name
        self.lr = lr
        self.epochs = epochs
        
        """weightを保存するディレクトリを決定する。"""
        # run/model_name/experiment_* を探す。
        self.directory = os.path.join('run', self.model_name)
        self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))
        
        # 実行結果をナンバリングするためにexperiment_* の数字を得る。experiment_*が無ければ 0
        run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0
        
        # experiment_(*+1) が無ければ作る。
        self.experiment_dir = os.path.join(self.directory, 'experiment_{:0=2}'.format(run_id))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        """checkpointを保存"""
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)
        
        """best modelの更新"""
        if is_best:
            # pred scoreを記録
            best_pred = state['best_pred']
            with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
                f.write(str(best_pred))
            # 過去のexperiment_*があるならそれとも比較
            if self.runs:
                previous_pred = [0.0]
                for run in self.runs:
                    run_id = run.split('_')[-1]
                    path = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)), 'best_pred.txt')
                    if os.path.exists(path):
                        with open(path, 'r') as f:
                            pred_score = float(f.readline())
                            previous_pred.append(pred_score)
                    else:
                        continue
                max_pred = max(previous_pred)
                if best_pred > max_pred:
                    shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))
            else:
                shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))

    def save_experiment_config(self):
        """
        学習時にハイパーパラメータのログを取る。
        ログ取りたい変数を好きに指定すること。
        """
        logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        log_file = open(logfile, 'w')
        p = OrderedDict()
        p['lr'] = self.lr
        p['epoch'] = self.epochs
        
        for key, val in p.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.close()