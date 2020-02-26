import numpy as np


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def Precision(self):
        PR = self.confusion_matrix[1,1] / np.sum(self.confusion_matrix[:,1])
        return PR

    def Recall(self):
        RC = self.confusion_matrix[1,1] / np.sum(self.confusion_matrix[1,:])
        return RC

    def F1_Score(self):
        PR = self.confusion_matrix[1,1] / np.sum(self.confusion_matrix[:,1])
        RC = self.confusion_matrix[1,1] / np.sum(self.confusion_matrix[1,:])
        F1 = 2*PR*RC/(PR+RC + 1e-6)
        return F1

    def _generate_matrix(self, gt_mask, pred_mask):
        mask = (gt_mask >= 0) & (gt_mask < self.num_class)
        label = self.num_class * gt_mask[mask].astype('int') + pred_mask[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_mask, pred_mask):
        assert gt_mask.shape == pred_mask.shape
        self.confusion_matrix += self._generate_matrix(gt_mask, pred_mask)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)




