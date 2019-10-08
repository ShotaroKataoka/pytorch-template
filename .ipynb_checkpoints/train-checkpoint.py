import os
import argparse

import numpy as np
import gensim
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Project Modules
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from dataloader import make_data_loader
from modeling.modeling import Modeling
from config import Config, pycolor
conf = Config()

class Trainer(object):
    def __init__(self, args):
        self.args = args
        
        # Define Utils. (No need to Change.)
        """
        These are Project Modules.
        You don't have to change these.
        
        Saver: To save model weight.  <utils.saver.Saver()>
        TensorboardSummary: To write tensorboard file.  <utils.summaries.TensorboardSummary()>
        Evaluator: To calculate some metrics (e.g. Accuracy).  <utils.metrics.Evaluator()>
        """
        ## ***Define Saver***
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        
        ## ***Define Tensorboard Summary***
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        
        ## ***Define Evaluator***
        self.evaluator = Evaluator(self.nclass)
        
        
        # Define Training components. (You have to Change!)
        """
        These are important setting for training.
        You have to change these.
        
        make_data_loader: This calls <dataloader>  <dataloader.__init__>
        Modeling: You have to define your model in <modeling.modeling.Modeling()> or another file.
        Optimizer: You have to define Optimizer.  (e.g. Adam, SGD)
        Criterion: You have to define Loss function. (e.g. CrossEntropy)
        """
        ## ***Define Dataloader***
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(self.args.batch_size)
        
        ## ***Define Your Model***
        model = Modeling(c_in=conf.input_channel,
                         c_out=conf.num_class,
                         c_hidden=conf.hidden_channel,
                         hidden_layer=conf.hidden_layer,
                         kernel_size=3)

        # ***Define Optimizer***
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr,
                                     weight_decay=args.weight_decay)
        
        # ***Define Criterion***
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.model, self.optimizer = model, optimizer
        
        

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            self.model = self.model.cuda()

        # Resuming checkpoint
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

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        self.evaluator.reset()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        print(pycolor.GREEN + "[Epoch: %d]" % (epoch) + pycolor.END)
        print(pycolor.YELLOW+"Training:"+pycolor.END)
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.criterion(output, target).sum()
            loss.backward()
            self.optimizer.step()
            train_loss += loss.sum().item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)
            # Compute Metrics
            pred = output.data.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            target = target.cpu().numpy()
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)
            
        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('Size of train set: %5d' % (i * self.args.batch_size + image.data.shape[0]))
        print('Total train loss: %.3f' % (train_loss / (i + 1)))
        Acc = self.evaluator.Accuracy()
        F_score_Average = self.evaluator.F_score_Average()
        print("Acc:{}, F_score:{}".format(Acc, F_score_Average))

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)


    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        print()
        print(pycolor.YELLOW+"Validation:"+pycolor.END)
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            test_loss += loss.sum().item()
            tbar.set_description('Validation loss: %.3f' % (test_loss / (i + 1)))
            # Compute Metrics
            pred = output.data.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            target = target.cpu().numpy()
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Accuracy()
        F_score_Average = self.evaluator.F_score_Average()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/F_score', F_score_Average, epoch)
        print('Size of validation set: %5d' % (i * self.args.batch_size + image.data.shape[0]))
        print('Total validation loss: %.3f' % (test_loss / (i + 1)))
        print("Acc:{}, F_score:{}".format(Acc, F_score_Average))
        print('---------------------')

        new_pred = F_score_Average
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

def main():
    parser = argparse.ArgumentParser(description="PyTorch Template.")
    parser.add_argument('--model-name', type=str, default='model01',
                        help='model name (default model01)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
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
        trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)

    trainer.writer.close()

if __name__ == "__main__":
    main()
