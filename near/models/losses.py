from typing import Optional, Literal, Tuple, Dict
import torch
import torch.nn.functional as F

def _flatten(logits, labels):
    """Flatten logits to [BP,K] and labels to [BP]."""
    if logits.dim() == 5:  # [B,K,D,H,W]
        B,K,D,H,W = logits.shape
        return logits.permute(0,2,3,4,1).reshape(B*D*H*W, K), labels.reshape(B*D*H*W)
    if logits.dim() == 3:  # [B,N,K] or [B,K,N]
        if logits.shape[1] != logits.shape[-1]:  # [B,N,K]
            B,N,K = logits.shape
            return logits.reshape(B*N, K), labels.reshape(B*N)
        else:                                    # [B,K,N]
            B,K,N = logits.shape
            return logits.permute(0,2,1).reshape(B*N, K), labels.reshape(B*N)
    raise ValueError("Unsupported logits shape")

def _one_hot(labels, K):
    if labels.dim() == 4:  # [B,D,H,W]
        return F.one_hot(labels.long(), K).permute(0,4,1,2,3).float()
    if labels.dim() == 2:  # [B,N]
        return F.one_hot(labels.long(), K).permute(0,2,1).float()
    raise ValueError("labels must be [B,D,H,W] or [B,N]")

def soft_dice_per_class(prob, labels_oh, eps=1e-6):
    """Per-class soft Dice over batch+space."""
    if prob.dim() == 5:
        p = prob.reshape(prob.shape[0], prob.shape[1], -1)
        g = labels_oh.reshape(labels_oh.shape[0], labels_oh.shape[1], -1)
    else:
        p, g = prob, labels_oh
    inter = (p * g).sum(dim=(0,2))
    denom = p.sum(dim=(0,2)) + g.sum(dim=(0,2))
    return (2*inter + eps)/(denom + eps)

def multiclass_dice(prob, labels, K, average="macro"):
    """macro/micro Dice."""
    labels_oh = _one_hot(labels, K)
    if average == "macro":
        return soft_dice_per_class(prob, labels_oh).mean()
    elif average == "micro":
        if prob.dim() == 5:
            p = prob.reshape(prob.shape[0], prob.shape[1], -1)
            g = labels_oh.reshape(labels_oh.shape[0], labels_oh.shape[1], -1)
        else:
            p, g = prob, labels_oh
        inter = (p * g).sum()
        denom = p.sum() + g.sum()
        return (2*inter + 1e-6)/(denom + 1e-6)
    else:
        raise ValueError

def cross_entropy_mc(logits, labels, class_weights=None, label_smoothing=0.0, ignore_index=None):
    """Multi-class CE (softmax inside)."""
    lg, lb = _flatten(logits, labels)
    return F.cross_entropy(lg, lb, weight=class_weights, label_smoothing=label_smoothing,
                           ignore_index=(-100 if ignore_index is None else ignore_index))

def latent_l2_penalty(z, reduce=True):
    """L2 penalty on latent codes (optional)."""
    v = z.pow(2).sum(dim=1).sqrt()
    return v.mean() if reduce else v

def combined_loss(logits, labels, lambda_dice=0.0, lambda_latent=0.0, z=None, num_classes=11,
                  class_weights=None, label_smoothing=0.0, ignore_index=None, dice_average="macro"):
    """CE + λ(1-Dice) + λ||z||_2."""
    ce = cross_entropy_mc(logits, labels, class_weights, label_smoothing, ignore_index)
    # prob for Dice/stat
    if logits.dim() == 5:
        prob = logits.softmax(1)
    else:
        prob = logits.softmax(-1).permute(0,2,1) if logits.shape[1] != num_classes else logits.softmax(1)
    if lambda_dice > 0:
        dice = multiclass_dice(prob, labels, num_classes, average=dice_average)
        dice_term = (1 - dice)
    else:
        dice = prob.new_tensor(0.0)
        dice_term = prob.new_tensor(0.0)
    z_l2 = latent_l2_penalty(z) if (lambda_latent > 0 and z is not None) else prob.new_tensor(0.0)
    loss = ce + lambda_dice * dice_term + lambda_latent * z_l2

    with torch.no_grad():
            dpc = soft_dice_per_class(prob, _one_hot(labels, num_classes))
    return loss, {"loss": loss.detach(), "ce": ce.detach(), "dice_macro": dice.detach(), "z_l2": z_l2.detach(),
                  "dice_per_class": dpc.detach()}