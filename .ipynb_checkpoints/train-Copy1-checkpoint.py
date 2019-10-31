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
from my_setting import pycolor

# set disable log of optuna
optuna.logging.disable_default_handler()

class Trainer(object):
    def __init__(self, batch_size=32, optimizer_name="Adam", lr=1e-3, weight_decay=1e-5,
                 epochs=200, model_name="model01", gpu_ids=None, resume=None, tqdm=None):
        """
        args:
            batch_size = (int) batch_size of training and validation
            lr = (float) learning rate of optimization
            weight_decay = (float) weight decay of optimization
            epochs = (int) The number of epochs of training
            model_name = (string) The name of training model. Will be folder name.
            gpu_ids = (List) List of gpu_ids. (e.g. gpu_ids = [0, 1]). Use CPU, if it is None. 
            resume = (Dict) Dict of some settings. (resume = {"checkpoint_path":PATH_of_checkpoint, "fine_tuning":True or False}). 
                     Learn from scratch, if it is None.
            tqdm = (tqdm Object) progress bar object. Set your tqdm please.
                   Don't view progress bar, if it is None.
        """
        # Set params
        self.batch_size = batch_size
        self.epochs = epochs
        self.start_epoch = 0
        self.use_cuda = (gpu_ids is not None) and torch.cuda.is_available
        self.tqdm = tqdm
        self.use_tqdm = tqdm is not None
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
        self.saver = Saver(model_name, lr, epochs)
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
        Evaluator: You have to define Evaluator. / <utils.metrics.Evaluator()>
        Optimizer: You have to define Optimizer. / <utils.optimizer.Optimizer()>
        Loss: You have to define Loss function. / <utils.loss.Loss()>
        """
        ## ***Define Dataloader***
        self.train_loader, self.val_loader, self.test_loader, self.num_classes = make_data_loader(batch_size)
        
        ## ***Define Your Model***
        self.model = Modeling(self.num_classes)
        
        ## ***Define Evaluator***
        self.evaluator = Evaluator(self.num_classes)
        
        ## ***Define Optimizer***
        self.optimizer = Optimizer(self.model.parameters(), optimizer_name=optimizer_name, lr=lr, weight_decay=weight_decay)
        
        ## ***Define Loss***
        self.criterion = Loss()
        
        # ------------------------- #
        # Some settings
        """
        You don't have to touch bellow code.
        
        Using cuda: Enable to use cuda if you want.
        Resuming checkpoint: You can resume training if you want.
        """
        ## ***Using cuda***
        if self.use_cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=gpu_ids).cuda()

        ## ***Resuming checkpoint***
        """You can ignore bellow code."""
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
            
    def _run_epoch(self, epoch, mode="train", leave_progress=True, use_optuna=False):
        """
        run training or validation 1 epoch.
        You don't have to change almost of this method.
        
        args:
            epoch = (int) How many epochs this time.
            mode = {"train" or "val"}
            leave_progress = {True or False} Can choose whether leave progress bar or not.
            use_optuna = {True or False} Can choose whether use optuna or not.
        
        Change point (if you need):
        - Evaluation: You can change metrics of monitoring.
        - writer.add_scalar: You can change metrics to be saved in tensorboard.
        """
        # ------------------------- #
        leave_progress = leave_progress and not use_optuna
        # Initializing
        epoch_loss = 0.0
        ## Set model mode & tqdm (progress bar; it wrap dataloader)
        assert (mode=="train") or (mode=="val"), "argument 'mode' can be 'train' or 'val.' Not {}.".format(mode)
        if mode=="train":
            data_loader = self.tqdm(self.train_loader, leave=leave_progress) if self.use_tqdm else self.train_loader
            self.model.train()
            num_dataset = len(self.train_loader)
        elif mode=="val":
            data_loader = self.tqdm(self.val_loader, leave=leave_progress) if self.use_tqdm else self.val_loader
            self.model.eval()
            num_dataset = len(self.val_loader)
        ## Reset confusion matrix of evaluator
        self.evaluator.reset()
        
        # ------------------------- #
        # Run 1 epoch
        for i, sample in enumerate(data_loader):
            ## ***Get Input data***
            inputs, target = sample["input"], sample["label"]
            if self.use_cuda:
                inputs, target = inputs.cuda(), target.cuda()
                
            ## ***Calculate Loss <Train>***
            if mode=="train":
                self.optimizer.zero_grad()
                output = self.model(inputs)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
            ## ***Calculate Loss <Validation>***
            elif mode=="val":
                with torch.no_grad():
                    output = self.model(inputs)
                loss = self.criterion(output, target)
            epoch_loss += loss.item()
            ## ***Report results***
            if self.use_tqdm:
                data_loader.set_description('{} loss: {:.3f}'.format(mode, epoch_loss / (i + 1)))
            ## ***Add batch results into evaluator***
            target = target.cpu().numpy()
            output = torch.argmax(output, axis=1).data.cpu().numpy()
            self.evaluator.add_batch(target, output)
            
        ## **********Evaluate Score**********
        """You can add new metrics! <utils.metrics.Evaluator()>"""
        Acc = self.evaluator.Accuracy()
        
        if not use_optuna:
            ## ***Save eval into Tensorboard***
            self.writer.add_scalar('{}/loss_epoch'.format(mode), epoch_loss / (i + 1), epoch)
            self.writer.add_scalar('{}/Acc'.format(mode), Acc, epoch)
            print('Total {} loss: {:.3f}'.format(mode, epoch_loss / num_dataset))
            print("{0} Acc:{1:.2f}".format(mode, Acc))
        
        # Return score to watch. (update checkpoint or optuna's objective)
        return Acc
    
    def run(self, leave_progress=True, use_optuna=False):
        """
        Run all epochs of training and validation.
        """
        for epoch in tqdm(range(self.start_epoch, self.epochs)):
            print(pycolor.GREEN + "[Epoch: {}]".format(epoch) + pycolor.END)
            
            ## ***Train***
            print(pycolor.YELLOW+"Training:"+pycolor.END)
            self._run_epoch(epoch, mode="train", leave_progress=leave_progress, use_optuna=use_optuna)
            ## ***Validation***
            print(pycolor.YELLOW+"Validation:"+pycolor.END)
            score = self._run_epoch(epoch, mode="val", leave_progress=leave_progress, use_optuna=use_optuna)
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

    ## ***Training hyper params***
    parser.add_argument('--model_name', type=str, default="model01", metavar='Name', help='model name (default: model01)')
    parser.add_argument('--epochs', type=int, default=30, metavar='int', help='number of epochs to train (default: auto)')
    parser.add_argument('--batch_size', type=int, default=4, metavar='int', help='input batch size for training (default: 4)')
    
    ## ***Optimizer params***
    parser.add_argument('--optimizer_name', type=str, default="Adam", metavar='Name', choices=["Adam", "SGD"], help='Optimizer name (default: Adam)')
    parser.add_argument('--lr', type=float, default=1e-6, metavar='float', help='learning rate (default: 1e-6)')
    parser.add_argument('--weight_decay', type=float, default=5e-4, metavar='float', help='w-decay (default: 5e-4)')
    
    ## ***cuda, seed***
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--gpu_ids', type=str, default='0', help='use which gpu to train, must be a comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='int', help='random seed (default: 1)')
    
    ## ***checking point***
    parser.add_argument('--resume_path', type=str, default=None, help='put the path to resuming file if you need')
    parser.add_argument('--fine_tuning', action='store_true', default=False, help='finetuning on a different dataset')
    
    args = parser.parse_args()
    
    # ------------------------- #
    # Set params
    ## ***cuda setting***
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only.\ne.g.) 0,1,2')
    else:
        gpu_ids = None
    
    resume = {"checkpoint_path":args.resume_path, "fine_tuning":args.fine_tuning} if args.resume_path is not None else None
    
    print(args)
    torch.manual_seed(args.seed)
    
    # ------------------------- #
    # Start Learning
    trainer = Trainer(batch_size=args.batch_size, optimizer_name=args.optimizer_name, lr=args.epochs, weight_decay=args.weight_decay, 
                      epochs=args.epochs, model_name=args.model_name, gpu_ids=gpu_ids, resume=resume, tqdm=tqdm)
    
    print('Starting Epoch:', trainer.start_epoch)
    print('Total Epoches:', trainer.epochs)
    print(trainer.model)
    trainer.run()
    
if __name__ == "__main__":
    main()

    
    