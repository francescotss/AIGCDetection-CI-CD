import torch
import torch.nn as nn
import torch.nn.functional as F

def loss_fn_kd(outputs, labels, teacher_outputs, KD_T=20, KD_alpha=0.5):
    KD_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs/KD_T,dim=1),
                             F.softmax(teacher_outputs/KD_T,dim=1) * KD_alpha*KD_T*KD_T) +\
        F.cross_entropy(outputs, labels) * (1. - KD_alpha)
    return KD_loss

def loss_clampping(loss:torch.Tensor, min_val, max_val):
    if loss != 0.0 and (torch.isinf(loss) or torch.isnan(loss)):
        loss = torch.zeros(1)
    if loss > 0.0:
        loss = torch.clamp(loss, min=min_val, max=max_val)
    return loss