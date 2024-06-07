import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


def dice_idx(label, pred, num_classes=4, plot=None):
    pred = F.softmax(pred, dim=1)
    pred = torch.argmax(pred, dim=1).squeeze(1)

    if plot:
        plt.subplot(1, 2, 1)
        plt.imshow(pred[1, 7, :, :].cpu())
        plt.title("Prediction")

        plt.subplot(1, 2, 2)
        plt.imshow(label[1, 7, :, :].cpu())
        plt.title("Label")
        plt.savefig("results/pred_" + str(plot) + ".svg")

    dice_list = list()
    present_dice_list = list()

    pred = pred.view(-1)
    label = label.view(-1)

    for sem_class in range(num_classes):
        pred_inds = (pred == sem_class)
        target_inds = (label == sem_class)
        if target_inds.long().sum().item() == 0:
            dice_now = float('nan')
        else:
            intersection_now = (pred_inds[target_inds]).long().sum().item()
            sum_now = pred_inds.long().sum().item() + target_inds.long().sum().item()
            dice_now = 2 * float(intersection_now) / float(sum_now)
            present_dice_list.append(dice_now)
        dice_list.append(dice_now)
    return np.mean(present_dice_list)


def mIOU(label, pred, num_classes=4, plot=None):
    pred = F.softmax(pred, dim=1)
    pred = torch.argmax(pred, dim=1).squeeze(1)

    if plot:
        plt.subplot(1, 2, 1)
        plt.imshow(pred[1, 7, :, :].cpu())
        plt.title("Prediction")

        plt.subplot(1, 2, 2)
        plt.imshow(label[1, 7, :, :].cpu())
        plt.title("Label")
        plt.savefig("results/pred_" + str(plot) + ".svg")

    iou_list = list()
    present_iou_list = list()

    pred = pred.view(-1)
    label = label.view(-1)
    # Note: Following for loop goes from 0 to (num_classes-1)
    # and ignore_index is num_classes, thus ignore_index is
    # not considered in computation of IoU.
    for sem_class in range(num_classes):
        pred_inds = (pred == sem_class)
        target_inds = (label == sem_class)
        if target_inds.long().sum().item() == 0:
            iou_now = float('nan')
        else:
            intersection_now = (pred_inds[target_inds]).long().sum().item()
            union_now = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection_now
            iou_now = float(intersection_now) / float(union_now)
            present_iou_list.append(iou_now)
        iou_list.append(iou_now)
    return np.mean(present_iou_list)