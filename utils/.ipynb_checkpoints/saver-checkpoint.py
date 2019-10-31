import os
import shutil
from collections import OrderedDict
import glob

import torch

class Saver(object):
    """
    This module is used to save model weights and hyper-param settings.
    
    __init__(): set determine directory to save files.
    save_checkpoint(): save weights of best model (every epochs.)
    save_experiment_config(): save hyper-param settings.
    """
    def __init__(self, model_name, lr, epochs):
        self.model_name = model_name
        self.lr = lr
        self.epochs = epochs
        
        # Determine directory where it save checkpoint.
        ## look for ./run/<model_name>/experiment_<num>
        self.directory = os.path.join('run', self.model_name)
        self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))
        
        ## Get <num> to determine next experiment.
        run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0
        
        ## Make directory "./run/<model_name>/experiment_(<num>+1)" if it is not exist.
        self.experiment_dir = os.path.join(self.directory, 'experiment_{:0=2}'.format(run_id))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

    def save_checkpoint(self, state, filename='checkpoint.pth.tar'):
        # Save checkpoint
        """
        1. Save checkpoint if it is best in this training.
        2. Save checkpoint if it is best in whole past experiments.
        """
        ## 1. Save checkpoint to "./run/<model_name>/experiment_(<num>+1)/checkpoint.pth.tar"
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)
        
        ## 2. Update best model to "./run/<model_name>/model_best.pth.tar"
        best_pred = state['best_pred']
        with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
            f.write(str(best_pred))
        # ***Compare with past experiments***
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
        Save hyper-param setting.
        
        You can add "params" to save like below.
        `p["hidden_dim"] = hidden_dim`
        """
        logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        log_file = open(logfile, 'w')
        p = OrderedDict()
        p['lr'] = self.lr
        p['epoch'] = self.epochs
        
        for key, val in p.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.close()