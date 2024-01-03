import torch



def IoU(gt, pred):
    '''
    Args:
    gt: ground truth segmentation mask
    pred: predict segmentation mask
    '''
    esp = 1e-10
    intsc = torch.sum(torch.logical_and(gt, pred).float())
    union = torch.sum(torch.logical_or(gt, pred).float())
    IoU = intsc / (union + esp)
    return IoU
    