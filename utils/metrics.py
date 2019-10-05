"""

"""
import numpy as np

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)
        
    def Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    # ↓これ精度じゃね？
    def Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc
    
    def Recall(self):
        Recall = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Recall[np.isnan(Recall)] = 0
        return Recall
    
    def Precision(self):
        Precision = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=0)
        Precision[np.isnan(Precision)] = 0
        return Precision
        
    def F_score(self):
        Recall = self.Recall()
        Precision = self.Precision()
        F_score = 2 * Recall * Precision / (Recall + Precision)
        F_score[np.isnan(F_score)] = 0
        return F_score
    
    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix
    
    def add_batch(self, target, pred):
        assert target.shape == pred.shape
        self.confusion_matrix += self._generate_matrix(target, pred)
    
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
