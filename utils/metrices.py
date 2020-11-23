import numpy as np
import torch
from sklearn.metrics import f1_score, average_precision_score
from sklearn.metrics import precision_recall_curve, roc_curve

SMOOTH = 1e-6
__all__ = ['get_f1_scores', 'get_ap_scores', 'batch_pix_accuracy', 'batch_intersection_union', 'get_iou', 'get_pr',
           'get_roc', 'get_ap_multiclass']


def get_iou(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    labels = labels.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    return iou.cpu().numpy()


def get_f1_scores(predict, target, ignore_index=-1):
    # Tensor process
    batch_size = predict.shape[0]
    predict = predict.data.cpu().numpy().reshape(-1)
    target = target.data.cpu().numpy().reshape(-1)
    pb = predict[target != ignore_index].reshape(batch_size, -1)
    tb = target[target != ignore_index].reshape(batch_size, -1)

    total = []
    for p, t in zip(pb, tb):
        total.append(np.nan_to_num(f1_score(t, p)))

    return total


def get_roc(predict, target, ignore_index=-1):
    target_expand = target.unsqueeze(1).expand_as(predict)
    target_expand_numpy = target_expand.data.cpu().numpy().reshape(-1)
    # Tensor process
    x = torch.zeros_like(target_expand)
    t = target.unsqueeze(1).clamp(min=0)
    target_1hot = x.scatter_(1, t, 1)
    batch_size = predict.shape[0]
    predict = predict.data.cpu().numpy().reshape(-1)
    target = target_1hot.data.cpu().numpy().reshape(-1)
    pb = predict[target_expand_numpy != ignore_index].reshape(batch_size, -1)
    tb = target[target_expand_numpy != ignore_index].reshape(batch_size, -1)

    total = []
    for p, t in zip(pb, tb):
        total.append(roc_curve(t, p))

    return total


def get_pr(predict, target, ignore_index=-1):
    target_expand = target.unsqueeze(1).expand_as(predict)
    target_expand_numpy = target_expand.data.cpu().numpy().reshape(-1)
    # Tensor process
    x = torch.zeros_like(target_expand)
    t = target.unsqueeze(1).clamp(min=0)
    target_1hot = x.scatter_(1, t, 1)
    batch_size = predict.shape[0]
    predict = predict.data.cpu().numpy().reshape(-1)
    target = target_1hot.data.cpu().numpy().reshape(-1)
    pb = predict[target_expand_numpy != ignore_index].reshape(batch_size, -1)
    tb = target[target_expand_numpy != ignore_index].reshape(batch_size, -1)

    total = []
    for p, t in zip(pb, tb):
        total.append(precision_recall_curve(t, p))

    return total


def get_ap_scores(predict, target, ignore_index=-1):
    total = []
    for pred, tgt in zip(predict, target):
        target_expand = tgt.unsqueeze(0).expand_as(pred)
        target_expand_numpy = target_expand.data.cpu().numpy().reshape(-1)

        # Tensor process
        x = torch.zeros_like(target_expand)
        t = tgt.unsqueeze(0).clamp(min=0).long()
        target_1hot = x.scatter_(0, t, 1)
        predict_flat = pred.data.cpu().numpy().reshape(-1)
        target_flat = target_1hot.data.cpu().numpy().reshape(-1)

        p = predict_flat[target_expand_numpy != ignore_index]
        t = target_flat[target_expand_numpy != ignore_index]

        total.append(np.nan_to_num(average_precision_score(t, p)))

    return total


def get_ap_multiclass(predict, target):
    total = []
    for pred, tgt in zip(predict, target):
        predict_flat = pred.data.cpu().numpy().reshape(-1)
        target_flat = tgt.data.cpu().numpy().reshape(-1)

        total.append(np.nan_to_num(average_precision_score(target_flat, predict_flat)))

    return total


def batch_precision_recall(predict, target, thr=0.5):
    """Batch Precision Recall
    Args:
        predict: input 4D tensor
        target: label 4D tensor
    """
    # _, predict = torch.max(predict, 1)

    predict = predict > thr
    predict = predict.data.cpu().numpy() + 1
    target = target.data.cpu().numpy() + 1

    tp = np.sum(((predict == 2) * (target == 2)) * (target > 0))
    fp = np.sum(((predict == 2) * (target == 1)) * (target > 0))
    fn = np.sum(((predict == 1) * (target == 2)) * (target > 0))

    precision = float(np.nan_to_num(tp / (tp + fp)))
    recall = float(np.nan_to_num(tp / (tp + fn)))

    return precision, recall


def batch_pix_accuracy(predict, target):
    """Batch Pixel Accuracy
    Args:
        predict: input 3D tensor
        target: label 3D tensor
    """

    # for thr in np.linspace(0, 1, slices):

    _, predict = torch.max(predict, 0)
    predict = predict.cpu().numpy() + 1
    target = target.cpu().numpy() + 1
    pixel_labeled = np.sum(target > 0)
    pixel_correct = np.sum((predict == target) * (target > 0))
    assert pixel_correct <= pixel_labeled, \
        "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(predict, target, nclass):
    """Batch Intersection of Union
    Args:
        predict: input 3D tensor
        target: label 3D tensor
        nclass: number of categories (int)
    """
    _, predict = torch.max(predict, 0)
    mini = 1
    maxi = nclass
    nbins = nclass
    predict = predict.cpu().numpy() + 1
    target = target.cpu().numpy() + 1

    predict = predict * (target > 0).astype(predict.dtype)
    intersection = predict * (predict == target)
    # areas of intersection and union
    area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all(), \
        "Intersection area should be smaller than Union area"
    return area_inter, area_union


# ref https://github.com/CSAILVision/sceneparsing/blob/master/evaluationCode/utils_eval.py
def pixel_accuracy(im_pred, im_lab):
    im_pred = np.asarray(im_pred)
    im_lab = np.asarray(im_lab)

    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    pixel_labeled = np.sum(im_lab > 0)
    pixel_correct = np.sum((im_pred == im_lab) * (im_lab > 0))
    # pixel_accuracy = 1.0 * pixel_correct / pixel_labeled
    return pixel_correct, pixel_labeled


def intersection_and_union(im_pred, im_lab, num_class):
    im_pred = np.asarray(im_pred)
    im_lab = np.asarray(im_lab)
    # Remove classes from unlabeled pixels in gt image.
    im_pred = im_pred * (im_lab > 0)
    # Compute area intersection:
    intersection = im_pred * (im_pred == im_lab)
    area_inter, _ = np.histogram(intersection, bins=num_class - 1,
                                 range=(1, num_class - 1))
    # Compute area union:
    area_pred, _ = np.histogram(im_pred, bins=num_class - 1,
                                range=(1, num_class - 1))
    area_lab, _ = np.histogram(im_lab, bins=num_class - 1,
                               range=(1, num_class - 1))
    area_union = area_pred + area_lab - area_inter
    return area_inter, area_union
