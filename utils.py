import numpy as np
from medpy.metric.binary import hd
np.bool = np.bool_ # fix for medpy using np.bool_ instead of bool

def dice_score(seg, gt, labels=[1,2,3,4,5,6]):
    '''
    Calculate dice score for given segmentation and ground truth for the given labels.
    Both seg and gt should be numpy arrays with the same shape in label (not one-hot) format.
    :param seg: predicted segmentation
    :param gt: ground truth segmentation
    :param labels: labels to calculate dice score for
    '''
    intersection = 0
    union = 0
    for k in labels:
        intersection += np.sum(seg[gt == k] == k) * 2.0
        union+=(np.sum(seg[seg == k] == k) + np.sum(gt[gt == k] == k))
    dice = intersection / union
    return dice


def create_visualization(ultrasound, segmentation,labels=[1,4,5,6,7,8,9],
                     colors=np.array([(1,0,0),(0,0,1),(0,1,0),(1,1,0),(0,1,1),(1,0,1),(1,1,1)])):
    '''
    Create visualization of segmentation on top of ultrasound image.
    :param ultrasound: ultrasound image
    :param segmentation: segmentation mask
    :param labels: labels to visualize
    :param colors: colors to use for visualization. There must be at least as many colors as labels. The order of the
                   colors should correspond to the order of the labels.
    :return: visualization of segmentation on top of ultrasound image as numpy array
    '''
    result = np.zeros((ultrasound.shape[0], ultrasound.shape[1], 3))
    for i in range(3):
        result[:, :, i] = ultrasound/255

    if len(labels)>len(colors):
        print('not enough colors for plotting')
        raise ValueError


    if len(segmentation.shape) == 3:
        print('one hot encoded tensor not implemented')
        raise NotImplementedError
    else:
        # Segmentation
        for i,label in enumerate(labels):
            if colors[i,0]!=0:
                result[segmentation == label, 0] = np.clip(colors[i,0]*(0.35 +
                                                     result[segmentation == label, 0]), 0.0, 1.0)
            if colors[i,1]!=0:
                result[segmentation == label, 1] = np.clip(colors[i,1]*(0.35 +
                                                     result[segmentation == label, 1]), 0.0, 1.0)
            if colors[i,2]!=0:
                result[segmentation == label, 2] = np.clip(colors[i,2]*(0.35 +
                                                     result[segmentation == label, 2]), 0.0, 1.0)
    return (result*255).astype(np.uint8)


def hausdorf(seg,gt,k=1):
    '''
    Calculate hausdorff distance for given segmentation and ground truth for the given label.
    Both seg and gt should be numpy arrays with the same shape in label (not one-hot) format.
    :param seg: predicted segmentation
    :param gt: ground truth segmentation
    :param k: label to calculate hausdorff distance for
    :return: hausdorff distance
    '''
    return hd(seg==k,gt == k)


if __name__ == '__main__':
    # quick test code
    seg = np.array([[0, 0, 1, 1, 0, 0],
                   [0, 0, 1, 1, 0, 0]])
    gt = np.array([[0, 0, 0, 1, 0, 0],
                   [0, 0, 1, 1, 0, 0]])
    print(dice_score(seg, gt,[1]))



