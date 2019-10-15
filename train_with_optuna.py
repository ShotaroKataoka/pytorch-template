import os
import argparse

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
from dataloader import make_data_loader
from modeling.modeling import Modeling
from config import Config, pycolor

# instance of config
conf = Config()
# set disable log of optuna
optuna.logging.disable_default_handler()

class Trainer(object):
    def __init__(self, args, trial=None):
        # ------------------------- #
        # Define Hyper-Params
        """
        You can choose how to optimize hyper-params (auto or manual.)
        If you set arg --optuna, hyper-params are optimized automatically.
        
        args: input value through command line.
        trial: input value of Optuna.
        """
        ## Get params
        self.args = args
        hyper_params = self.define_hyper_params(trial)
        
        ## Train param
        batch_size = hyper_params["batch_size"]
        epochs = args.epochs
        
        ## Optimizer param
        optimizer_name = hyper_params["optimizer_name"]
        lr = hyper_params["lr"]
        weight_decay = hyper_params["weight_decay"]
        
        # ------------------------- #
        # Define Utils. (No need to Change.)
        """
        These are Project Modules.
        You may not have to change these.
        
        Saver: To save model weight.  <utils.saver.Saver()>
        TensorboardSummary: To write tensorboard file.  <utils.summaries.TensorboardSummary()>
        Evaluator: To calculate some metrics (e.g. Accuracy).  <utils.metrics.Evaluator()>
        """
        ## ***Define Saver***
        self.saver = Saver(args.model_name, lr, epochs)
        self.saver.save_experiment_config()
        
        ## ***Define Tensorboard Summary***
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        
        # ------------------------- #
        # Define Training components. (You have to Change!)
        """
        These are important setting for training.
        You have to change these.
        
        make_data_loader: This creates some <Dataloader>s.  <dataloader.__init__>
        Modeling: You have to define your model in <modeling.modeling.Modeling()> or another file.
        Optimizer: You have to define Optimizer.  (e.g. Adam, SGD)
        Criterion: You have to define Loss function. (e.g. CrossEntropy)
        """
        ## ***Define Dataloader***
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(self.args.batch_size)
        
        ## ***Define Your Model***
        model = Modeling(c_in=conf.input_channel,
                         c_out=conf.num_class,
                         c_hidden=conf.hidden_channel,
                         hidden_layer=conf.hidden_layer,
                         kernel_size=3)
        
        ## ***Define Evaluator***
        self.evaluator = Evaluator(self.nclass)
        
        ## ***Define Optimizer***
        if optimizer_name=="Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name=="SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        ## ***Define Criterion***
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.model, self.optimizer = model, optimizer
        
        # ------------------------- #
        # Some settings
        """
        You don't have to touch bellow code.
        
        Using cuda: Enable to use cuda if you want.
        Resuming checkpoint: You can resume training if you want.
        Clear start epoch if fine-tuning: fine tuning setting.
        """
        ## ***Using cuda***
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            self.model = self.model.cuda()

        ## ***Resuming checkpoint***
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        ## ***Clear start epoch if fine-tuning***
        if args.ft:
            args.start_epoch = 0

    def run(self, epoch, mode="train", optuna=False):
        """
        run training or validation 1 epoch.
        You don't have to change almost of this method.
        
        Change point (if you need):
        - Evaluation: You can change metrics of monitoring.
        - writer.add_scalar: You can change metrics to be saved tensorboard.
        """
        # ------------------------- #
        # Initializing
        epoch_loss = 0.0
        ## Set model mode & tqdm (progress bar; it wrap dataloader)
        assert mode=="train" or mode=="val", "argument 'mode' can be 'train' or 'val.' Not {}.".format(mode)
        if mode=="train":
            print(pycolor.GREEN + "[Epoch: {}]".format(epoch) + pycolor.END)
            print(pycolor.YELLOW+"Training:"+pycolor.END)
            self.model.train()
            tbar = tqdm(self.train_loader)
            num_dataset = len(self.train_loader)
        elif mode=="val":
            print(pycolor.YELLOW+"Validation:"+pycolor.END)
            self.model.eval()
            tbar = tqdm(self.val_loader)
            num_dataset = len(self.val_loader)
        ## Reset confusion matrix of evaluator
        self.evaluator.reset()
        
        # ------------------------- #
        # Run 1 epoch
        for i, sample in enumerate(tbar):
            inputs, target = sample["image"], sample["label"]
            if self.args.cuda:
                inputs, target = inputs.cuda(), target.cuda()
            if mode=="train":
                self.optimizer.zero_grad()
                output = self.model(inputs)
            elif mode=="val":
                with torch.no_grad():
                    output = self.model(inputs)
            loss = self.criterion(output, target).sum()
            if mode=="train":
                loss.backward()
                self.optimizer.step()
            epoch_loss += loss.item()
            tbar.set_description('{} loss: {:.3f}'.format(mode, (epoch_loss / ((i + 1)*self.args.batch_size))))
            # Compute Metrics
            pred = output.data.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            target = target.cpu().numpy()
            ## Add batch into evaluator
            self.evaluator.add_batch(target, pred)
            
        # ------------------------- #
        # Save Log
        ## **********Evaluate**********
        Acc = self.evaluator.Accuracy()
        F_score_Average = self.evaluator.F_score_Average()
        ## Save results
        self.writer.add_scalar('{}/loss_epoch'.format(mode), epoch_loss / num_dataset, epoch)
        self.writer.add_scalar('{}/Acc'.format(mode), Acc, epoch)
        self.writer.add_scalar('{}/F_score'.format(mode), F_score_Average, epoch)
        print('Total {} loss: {:.3f}'.format(mode, epoch_loss / num_dataset))
        print("Acc:{}, F_score:{}".format(Acc, F_score_Average))
        ## Save model
        is_save = False
        if mode=="train" and self.args.no_val:
            is_best = False
            is_save = True
        elif mode=="val":
            new_pred = F_score_Average
            print("---------------------")
            if new_pred > self.best_pred:
                is_best = True
                is_save = True
                print("model improve best score from {:.4f} to {:.4f}.".format(self.best_pred, new_pred))
                self.best_pred = new_pred
        if is_save:
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

    def define_hyper_params(self, trial):
        """
        This method define hyper-params by args or Optuna.
        If you set arg "--optuna", this method offer hyper-params optimized by Optuna.
        
        Optuna is tool for optimizing hyper-params using bayesian method.
        Although it can optimize hyper-params automatically, it is very heavy.
        """
        if self.args.optuna:
            """
            If you use Optuna, you can add hyper-params which you want to optimize.
            https://optuna.readthedocs.io/en/latest/tutorial/configurations.html
            """
            ## Train param
            batch_size = trial.suggest_categorical('batch_size', [1, 2, 4, 8, 16, 32, 64])
            ## Optimizer param
            optimizer_name = trial.suggest_categorical('optimizer', ["Adam", "SGD"])
            lr = trial.suggest_loguniform('lr', 1e-8, 1e-2)
            weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)
        else:
            ## Train param
            batch_size = self.args.batch_size
            ## Optimizer param
            optimizer_name = self.args.optimizer_name
            lr = self.args.lr
            weight_decay = self.args.weight_decay
            
        return {"lr": lr,
                "batch_size": batch_size,
                "optimizer_name": optimizer_name,
                "weight_decay": weight_decay}
        
        
def main():
    parser = argparse.ArgumentParser(description="PyTorch Template.")
    parser.add_argument('--model-name', type=str, default='model01',
                        help='model name (default model01)')
    parser.add_argument('--optimizer_name', type=str, default='Adam',
                        help='optimizer name (default Adam)')
    parser.add_argument('--optuna', action='store_true', default=
                        False, help='use Optuna')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=1e-6, metavar='LR',
                        help='learning rate (default: 1e-6)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    args.lr = args.lr / (4 * len(args.gpu_ids)) * args.batch_size

    if args.checkname is None:
        args.checkname = 'default-model'
    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.run(epoch, mode="train")
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.run(epoch, mode="val")

    trainer.writer.close()

if __name__ == "__main__":
    main()

    
    