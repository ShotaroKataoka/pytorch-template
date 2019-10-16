import numpy as np
import gensim
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import os

# Original Modules
from dataloader import make_data_loader
from utils.metrics import Evaluator
from modeling.modeling import Modeling
from config import Config, pycolor
conf = Config()

class Predictor(object):
    def __init__(self, PATH):
        # Define Dataloader
        # word_vector = gensim.models.KeyedVectors.load_word2vec_format(conf.word_vector_dir+'model.vec', binary=False)
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(16)
        
        print(pycolor.CYAN+"  Define Model."+pycolor.END)
        # Define network (****Change****)
        model = Modeling(embedding_dim=conf.embedding_dim,
                         c_out=conf.num_class,
                         c_hidden=conf.hidden_channel,
                         hidden_layer=conf.hidden_layer)

        model_state = torch.load(PATH)
        state_dict = model_state["state_dict"]
        print("epoch: {}".format(model_state["epoch"]))
        
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        # load params
        model.load_state_dict(new_state_dict)
        
        # Define Criterion
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        
        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        
        # Using cuda
        if True:
            #model = torch.nn.DataParallel(model, device_ids=[0])
            model = model.cuda()

        self.model = model
        self.predicts = []
        self.answers = []
        
    def predict(self):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.test_loader, desc='\r')
        print()
        print(pycolor.YELLOW+"Test:"+pycolor.END)
        test_loss = 0.0
        self.predicts = []
        self.answers = []
        for i, sample in enumerate(tbar):
            question, target = sample['question'], sample['label']
            if True:
                question, target = question.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(question)
            loss = self.criterion(output, target)
            test_loss += loss.sum().item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            # Compute Metrics
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)
            
            # Count pred
            self.predicts += list(np.argsort(pred)[:,::-1])
            self.answers += list(target)

        # Fast test during the training
        self.Acc = self.evaluator.Accuracy()
        self.Top3Acc = self.evaluator.TopN_Accuracy()
        self.MRR = self.evaluator.MRR()
        