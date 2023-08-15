import torch.nn as nn
import torch
from torch.nn import functional as F

class SegCELoss(nn.Module):
    def __init__(self):
        super(SegCELoss, self).__init__()
        weight = torch.FloatTensor([0.9, 1.1]).cuda()
        self.seg_criterion = nn.CrossEntropyLoss(weight=weight)
        
    def forward(self, pred, targets):
        '''
            pred: [BxKxhxw]
            targets['mask']: [BxHxW]
        '''
        target = targets['mask']
        if pred.shape[-2:] != target.shape[-2:]:
            h, w = target.size(1), target.size(2)
            pred = F.interpolate(input=pred, size=(h, w), mode='bilinear', align_corners=True)
        seg_loss = self.seg_criterion(pred, target)
        loss_dict = {'total_loss':seg_loss}
            
        return loss_dict

criterion_dict = {
    'caris': SegCELoss,
}