import torch
import torch.nn as nn

class SoftDiceLoss(nn.Module):
    """
    From: https://github.com/neptune-ml/open-solution-mapping-challenge/blob/master/src/steps/pytorch/validation.py 
    """
    def __init__(self, smooth=0, eps = 1e-6):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, logit, target):
        return 1 - (2 * torch.sum(logit * target) + self.smooth) / (
                    torch.sum(logit) + torch.sum(target) + self.smooth + self.eps)

class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce','focal','wce','dice']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode == 'wce':
            return self.WCrossEntropyLoss
        elif mode == 'dice':
            return self.DiceLoss
        elif mode == 'dicece':
            return self.DiceCrossEntropyLoss
        elif mode == 'dicewce':
            return self.DiceWCrossEntropyLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        target = target[:,0,:,:]
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def WCrossEntropyLoss(self, logit, target, dist_w):
        n, c, h, w = logit.size()
        target = target[:,0,:,:]
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average,reduce=False)
        if self.cuda:
            criterion = criterion.cuda()

        loss_per_pixel = criterion(logit, target.long())
        loss = torch.mean(loss_per_pixel * dist_w.float())

        if self.batch_average:
            loss /= n

        return loss


    def DiceLoss(self, logit, target, smooth = 0, eps = 1e-6, excluded_classes=[]):
        """Calculate Dice Loss for multiple class output.
        Adapted from: https://github.com/neptune-ml/open-solution-mapping-challenge/blob/master/src/models.py
        Args:
        logit (torch.Tensor): Model output of shape (N x C x H x W).
        target (torch.Tensor): Target of shape (N x 1 x H x W).
        smooth (float, optional): Smoothing factor. Defaults to 0.
        excluded_classes (list, optional):
            List of excluded classes numbers. Dice Loss won't be calculated
            against these classes. Often used on background when it has separate output class.
            Defaults to [].
        Returns:
        torch.Tensor: Loss value.
        """
        n, c, h, w = logit.size()
        target = target[:,0,:,:]
        activation_nn = torch.nn.Softmax2d()
        loss = 0
        softdice = SoftDiceLoss(smooth=smooth , eps = eps)
        logit = activation_nn(logit)

        for class_nr in range(logit.size(1)):
            if class_nr in excluded_classes:
                continue
            class_target = (target == class_nr)
            class_target.data = class_target.data.float()
            loss += softdice(logit[:, class_nr, :, :], class_target)

        if self.batch_average:
            loss /= n

        return loss

    def DiceWCrossEntropyLoss(self, logit, target, dist_w, 
                              dice_weight=0.5, smooth = 0, eps = 1e-6, excluded_classes=[],
                              wce_weight=0.5):
        """Calculate mixed Dice and Cross Entropy Loss.
        Adapted from: https://github.com/neptune-ml/open-solution-mapping-challenge/blob/master/src/models.py
        Args:
        logit (torch.Tensor): Model output of shape (N x C x H x W).
        target (torch.Tensor): Target of shape (N x 1 x H x W).
        dist_w (torch.Tensor): Distance weight of shape (N x 1 x H x W).
        dice_weight (float, optional): Weight of Dice loss. Defaults to 0.5.
        smooth (float, optional): Smoothing factor. Defaults to 0.
        excluded_classes (list, optional):
            List of excluded classes numbers. Dice Loss won't be calculated
            against these classes. Often used on background when it has separate output class.
            Defaults to [].
        wce_weight (float, optional): Weight of Cross Entropy loss. Defaults to 0.5.
        Returns:
        torch.Tensor: Loss value.
        """
        loss = dice_weight * self.DiceLoss(logit, target) + wce_weight * self.WCrossEntropyLoss(logit, target, dist_w)


        return loss

    def DiceCrossEntropyLoss(self, logit, target,  
                              dice_weight=0.5, smooth = 0, eps = 1e-6, excluded_classes=[],
                              ce_weight=0.5):
        """Calculate mixed Dice and Cross Entropy Loss.
        Adapted from: https://github.com/neptune-ml/open-solution-mapping-challenge/blob/master/src/models.py
        Args:
        logit (torch.Tensor): Model output of shape (N x C x H x W).
        target (torch.Tensor): Target of shape (N x 1 x H x W).
        dice_weight (float, optional): Weight of Dice loss. Defaults to 0.5.
        smooth (float, optional): Smoothing factor. Defaults to 0.
        excluded_classes (list, optional):
            List of excluded classes numbers. Dice Loss won't be calculated
            against these classes. Often used on background when it has separate output class.
            Defaults to [].
        wce_weight (float, optional): Weight of Cross Entropy loss. Defaults to 0.5.
        Returns:
        torch.Tensor: Loss value.
        """
        loss = dice_weight * self.DiceLoss(logit, target) + ce_weight * self.CrossEntropyLoss(logit, target)


        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        target = target[:,0,:,:]
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)

        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)

        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss

if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())




