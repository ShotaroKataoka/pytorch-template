import os
import argparse
from glob import glob

import numpy as np
import gensim
import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
from tqdm import tqdm

# Project Modules
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from utils.optimizer import Optimizer
from utils.loss import Loss
from dataloader import make_data_loader
from modeling.modeling import Modeling
from config import Config, pycolor

# instance of config
conf = Config()
# set disable log of optuna
optuna.logging.disable_default_handler()

class Trainer(object):
    def __init__(self, batch_size=32, epochs=200, lr=1e-3, weight_decay=1e-5,
                 gpu_ids=None, resume=None, tqdm=None):
        """
        batch_size : batch_size of training and validation
        epochs : epochs of training
        lr : learning rate of optimization
        weight_decay : weight decay of optimization
        gpu_ids : List of gpu_ids. (e.g. gpu_ids = [0, 1]). Use CPU, if it is None. 
        resume : Dict of some settings. (resume = {"checkpoint_path":PATH_of_checkpoint, "fine_tuning":True or False}). 
                 Learn from scratch, if it is None.
        tqdm : progress bar object. Set your tqdm please. 
        """
        self.use_cuda = (gpu_ids is not None) and torch.cuda.is_available
        self.batch_size = batch_size
        self.epochs = epochs
        self.tqdm = tqdm
        # ------------------------- #
        # Define Utils. (No need to Change.)
        """
        These are Project Modules.
        You may not have to change these.
        
        Saver: Save model weight. / <utils.saver.Saver()>
        TensorboardSummary: Write tensorboard file. / <utils.summaries.TensorboardSummary()>
        Evaluator: Calculate some metrics (e.g. Accuracy). / <utils.metrics.Evaluator()>
        """
        ## ***Define Saver***
        self.saver = Saver(self.args.model_name, lr, epochs)
        self.saver.save_experiment_config()
        
        ## ***Define Tensorboard Summary***
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        
        # ------------------------- #
        # Define Training components. (You have to Change!)
        """
        These are important setting for training.
        You have to change these.
        
        make_data_loader: This creates some <Dataloader>s. / <dataloader.__init__>
        Modeling: You have to define your Model. / <modeling.modeling.Modeling()>
        Optimizer: You have to define Optimizer. / <utils.optimizer.Optimizer()>
        Criterion: You have to define Loss function. / <utils.loss.Loss()>
        """
        ## ***Define Dataloader***
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(batch_size)
        
        ## ***Define Your Model***
        self.model = Modeling(c_in=conf.input_channel, c_out=conf.num_class, c_hidden=conf.hidden_channel, hidden_layer=conf.hidden_layer, kernel_size=3)
        
        ## ***Define Evaluator***
        self.evaluator = Evaluator(self.nclass)
        
        ## ***Define Optimizer***
        self.optimizer = Optimizer(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        ## ***Define Loss***
        self.criterion = Loss("Adam")
        
        # ------------------------- #
        # Some settings
        """
        You don't have to touch bellow code.
        
        Using cuda: Enable to use cuda if you want.
        Resuming checkpoint: You can resume training if you want.
        """
        ## ***Using cuda***
        if self.use_cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=gpu_ids)
            self.model = self.model.cuda()

        ## ***Resuming checkpoint***
        self.best_pred = 0.0
        if resume is not None:
            if not os.path.isfile(resume["checkpoint_path"]):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(resume["checkpoint_path"]))
            checkpoint = torch.load(resume["checkpoint_path"])
            self.start_epoch = checkpoint['epoch']
            if self.use_cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if resume["fine_tuning"]:
                # resume params of optimizer, if run fine tuning.
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.start_epoch = 0
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume["checkpoint_path"], checkpoint['epoch']))
            
    def _run_epoch(self, epoch, mode="train", leave_progress=True):
        """
        run training or validation 1 epoch.
        You don't have to change almost of this method.
        
        Change point (if you need):
        - Evaluation: You can change metrics of monitoring.
        - writer.add_scalar: You can change metrics to be saved in tensorboard.
        """
        # ------------------------- #
        # Initializing
        epoch_loss = 0.0
        ## Set model mode & tqdm (progress bar; it wrap dataloader)
        assert (mode=="train") or (mode=="val"), "argument 'mode' can be 'train' or 'val.' Not {}.".format(mode)
        if mode=="train":
            if self.tqdm is not None:
                tbar = self.tqdm(self.train_loader, leave=leave_progress)
            else:
                tbar = self.train_loader
            self.model.train()
            num_dataset = len(self.train_loader)
        elif mode=="val":
            if self.tqdm is not None:
                tbar = self.tqdm(self.val_loader, leave=leave_progress)
            else:
                tbar = self.val_loader
            self.model.eval()
            num_dataset = len(self.val_loader)
        ## Reset confusion matrix of evaluator
        self.evaluator.reset()
        
        # ------------------------- #
        # Run 1 epoch
        for i, sample in enumerate(tbar):
            inputs, target = sample["input"], sample["label"]
            if self.use_cuda:
                inputs, target = inputs.cuda(), target.cuda()
            if mode=="train":
                self.optimizer.zero_grad()
                output = self.model(inputs)
            elif mode=="val":
                with torch.no_grad():
                    output = self.model(inputs)
            loss = self.criterion(output, target)
            if mode=="train":
                loss.backward()
                self.optimizer.step()
            epoch_loss += loss.item()
            if self.tqdm is not None:
                tbar.set_description('{} loss: {:.3f}'.format(mode, (epoch_loss / ((i + 1)*self.batch_size))))
            # Compute Metrics
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            ## Add batch into evaluator
            self.evaluator.add_batch(target, pred)
            
        # ------------------------- #
        # Save Log
        ## **********Evaluate**********
        Acc = self.evaluator.Accuracy()
        
        ## Save eval
        self.writer.add_scalar('{}/loss_epoch'.format(mode), epoch_loss / num_dataset, epoch)
        self.writer.add_scalar('{}/Acc'.format(mode), Acc, epoch)
        print('Total {} loss: {:.3f}'.format(mode, epoch_loss / num_dataset))
        print("Acc:{}".format(Acc))
        
        return Acc
    
    def run(self):
        for epoch in tqdm(range(self.start_epoch, self.epochs)):
            print(pycolor.GREEN + "[Epoch: {}]".format(epoch) + pycolor.END)
            
            ## ***Train***
            print(pycolor.YELLOW+"Training:"+pycolor.END)
            self._run_epoch(epoch, mode="train", leave_progress=True)
            
            ## ***Validation***
            print(pycolor.YELLOW+"Validation:"+pycolor.END)
            score = trainer._run_epoch(epoch, mode="val", leave_progress=True)
            print("---------------------")
            if score > self.best_pred:
                print("model improve best score from {:.4f} to {:.4f}.".format(self.best_pred, score))
                self.best_pred = score
                self.saver.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'best_pred': self.best_pred,
                })
        self.writer.close()
        return self.best_pred
    
def main():
    # ------------------------- #
    # Set parser
    parser = argparse.ArgumentParser(description="PyTorch Template.")
    
    ## ***Optuna option***
#     parser.add_argument('--optuna', action='store_true', default=False, help='use Optuna')
#     parser.add_argument('--prune', action='store_true', default=False, help='use Optuna Pruning')
#     parser.add_argument('--trial_size', type=int, default=100, metavar='N', help='number of trials to optimize (default: 100)')
    
    ## ***Training hyper params***
    parser.add_argument('--model_name', type=str, default="model01", metavar='Name', help='model name (default: model01)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N', help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0, metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch_size', type=int, default=None, metavar='N', help='input batch size for training (default: auto)')
    
    ## ***Optimizer params***
    parser.add_argument('--lr', type=float, default=1e-6, metavar='LR', help='learning rate (default: 1e-6)')
    parser.add_argument('--weight_decay', type=float, default=5e-4, metavar='M', help='w-decay (default: 5e-4)')
    
    ## ***cuda, seed and logging***
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--gpu_ids', type=str, default='0', help='use which gpu to train, must be a comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    
    ## ***checking point***
    parser.add_argument('--resume_path', type=str, default=None, help='put the path to resuming file if needed')
    parser.add_argument('--fine_tuning', action='store_true', default=False, help='finetuning on a different dataset')
    
    args = parser.parse_args()
    
    # ------------------------- #
    # Set params
    ## ***cuda setting***
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')
        
    ## ***default batch_size setting***
    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    ## ***Model name***
    if args.optuna:
        directory = os.path.join('run', args.model_name+"_optuna_*")
        run = sorted(glob(directory))
        run_id = int(run[-1].split('_')[-1]) + 1 if run else 0
        args.model_name = args.model_name + "_optuna_{:0=2}".format(run_id)
        
    print(args)
    torch.manual_seed(args.seed)
    
    # ------------------------- #
    # Start Learning
    print('Starting Epoch:', args.start_epoch)
    print('Total Epoches:', args.epochs)
    
    if args.optuna:
        ## ***Use Optuna***
        TRIAL_SIZE = args.trial_size
        with tqdm(total=TRIAL_SIZE) as pbar:
            study = optuna.create_study()
            study.optimize(create_all_epochs_runner(args, pbar), n_trials=TRIAL_SIZE)
        ### Save study
        df = study.trials_dataframe()
        directory = os.path.join('run', args.model_name)
        df.to_csv(os.path.join(directory, "trial.csv"))
    else:
        ## ***Not use Optuna***
        train_runner = create_all_epochs_runner(args, None)
        train_runner(None)
    
if __name__ == "__main__":
    main()

    
    