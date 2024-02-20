import numpy as np
from medpy.metric.binary import hd
import matplotlib.pyplot as plt
import os
from PIL import Image
from skimage.transform import resize
import utils
import torch
np.bool = np.bool_ # fix for medpy using np.bool_ instead of bool

#### evaluation utils ####

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


def create_visualization(ultrasound, segmentation,labels=[1,2,3,4,5,6],
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


def resize_image(numpy_img,resize_dim=(256,256),annotation=False,convert_to_png=True):
    '''
    Resize numpy image to given dimensions
    :param numpy_img: numpy image to resize
    :param resize_dim: dimensions to resize to
    :param annotation: whether the image is an annotation or not. If True, the image is converted to one-hot encoding
                       before resizing, and converted back to an annotation after resizing to avoid rounding artefacts.
    :param convert_to_png: whether to convert the resized numpy image to a png image or not
    :return: resized numpy image or resized png image
    '''
    if annotation:
        # convert labels to one-hot encoding
        # this avoids problems with resizing and rounding errors creating artefacts
        numpy_img_one_hot=np.zeros((numpy_img.shape[0],numpy_img.shape[1],4))
        for row in range(numpy_img.shape[0]):
            for col in range(numpy_img.shape[1]):
                numpy_img_one_hot[row,col,int(numpy_img[row,col])]=1
        #resize each channel separately
        numpy_image_resized=np.zeros((resize_dim[0],resize_dim[1],4))
        for i in range(1,4):
            numpy_image_resized[:,:,i]=np.round(resize(numpy_img_one_hot[:,:,i],resize_dim))
        # recombine to one hot encoding
        numpy_image_resized=np.argmax(numpy_image_resized,axis=2).astype(np.uint8)
    else:
        numpy_image_resized=resize(numpy_img,resize_dim)
    if convert_to_png:
        img_data = Image.fromarray(numpy_image_resized)
        img_data_grayscale = img_data.convert("L")
        return img_data_grayscale
    else:
        return numpy_image_resized

def boxplot(metric_values,save_dir,title,ylabel,xticks,save_name,show=True):
    '''
    Create boxplot of given metric values and save it to save_dir. The metric can be for example dice scores or
    hausdorff distances.
    :param metric_values: list of metric scores. Each list element is a three element list with dice scores for each of the
                        three class, i.e. [[metric_lv,metric_myo,metric_la],[metric_lv,metric_myo,metric_la],...]
    :param save_dir: directory to save plot to
    :param title: title of plot
    :param ylabel: y-axis label
    :param xticks: x-axis labels
    :param show: whether to show plot or not
    '''
    # create boxplot of dice scores
    fig,ax=plt.subplots()
    ax.boxplot(metric_values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(xticks)
    # set limit of y-axis to 0-1
    ax.set_ylim([0,1])
    # remove whitespace
    fig.tight_layout()
    # save plot
    fig.savefig(os.path.join(save_dir,save_name))
    if show:
        plt.show()

def plot_segmentation(us_image,anno,pred,sample_name,dices,plot_folder,show=False):
    '''
    Plot annotation and prediction of a single sample
    :param us_image: ultrasound image
    :param anno: annotation of segmentation masks ('ground truth')
    :param pred: prediction by model
    :param sample_name: name of sample
    :param dices: dice scores of prediction compared to annotation. This is a list with dice scores for each of the
                  three class, i.e. [dice_lv,dice_myo,dice_la]
    :param plot_folder: folder to save plot to
    :param show: whether to show plot or not
    '''
    # plot anno and pred
    fig, ax = plt.subplots(1, 2)
    # visualization paints the segmentation on top of the ultrasound image
    visual_anno = utils.create_visualization(us_image, anno, labels=[1, 2, 3],
                                             colors=np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1)]))
    ax[0].imshow(visual_anno)
    visual_pred = utils.create_visualization(us_image, pred, labels=[1, 2, 3],
                                             colors=np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1)]))
    ax[1].imshow(visual_pred)
    # set titles
    ax[0].set_title('Annotation')
    ax[1].set_title('Prediction')
    dice_lv, dice_myo, dice_la = dices
    # set main title
    fig.suptitle(sample_name[:-4] + '\nDice LV: ' + str(np.round(dice_lv, 2)) +
                 ', Dice Myo: ' + str(np.round(dice_myo, 2)) +
                 ', Dice LA: ' + str(np.round(dice_la, 2)))
    # remove axis
    ax[0].axis('off')
    ax[1].axis('off')
    # remove whitespace
    fig.tight_layout()
    # save plot
    fig.savefig(os.path.join(plot_folder, sample_name))
    if show:
        plt.show()
    plt.close(fig)

#### preprocessing utils ####

def text_to_set(text_loc):
    '''
    Convert text file to set with one element per line
    :param text_loc: location of text file
    :return: set with one element per line
    '''
    with open(text_loc) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    return set(content)

def get_splits(split_nb,splits_loc):
    '''
    Get train, validation and test splits for a given split number
    :param split_nb: split number
    :param splits_loc: location of splits. This folder should have the following structure:
                       |-subgroups_CAMUS
                       | |-subGroup0_testing.txt
                       | |-subGroup0_training.txt
                       | |-subGroup0_validation.txt
                       | |-subGroup1_testing.txt
                       | |-subGroup1_training.txt
                       | |-subGroup1_validation.txt
                       | |-...
    :return: train, validation and test splits
    '''
    train_loc = os.path.join(str(splits_loc), f'subGroup{split_nb}_training.txt')
    val_loc = os.path.join(str(splits_loc), f'subGroup{split_nb}_validation.txt')
    test_loc = os.path.join(str(splits_loc), f'subGroup{split_nb}_testing.txt')
    train_set = text_to_set(train_loc)
    val_set = text_to_set(val_loc)
    test_set = text_to_set(test_loc)
    return train_set,val_set,test_set

def write_list_to_txt(list_of_strings,out_loc):
    '''
    Write list of strings to text file. Each string is written on a separate line.
    :param list_of_strings: list of strings
    :param out_loc: location of output text file
    '''
    if not os.path.exists(os.path.dirname(out_loc)):
        os.makedirs(os.path.dirname(out_loc))
    with open(out_loc,'w') as f:
        for item in list_of_strings:
            f.write("%s\n" % item)


#### training utils ####

def set_up_gpu(cuda_id=0,verbose=True):
    '''
    Return device for torch to use.
    :param cuda_id: cuda id of gpu to use
    :param verbose: whether to print info or not
    :return: device parameter for pytorch to use
    '''
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print('Using device:', device)
    return device

def convert_to_one_hot(seg,nb_classes=4,device='cpu'):
    '''
    Convert segmentation mask to one-hot encoding
    :param seg: segmentation mask
    :param nb_classes: number of classes including background
    :return: one-hot encoded segmentation mask
    '''
    seg_cpu=seg.cpu()
    seg_one_hot=np.zeros((seg_cpu.shape[0],nb_classes,seg_cpu.shape[1],seg_cpu.shape[2]))
    for i in range(nb_classes):
        seg_one_hot[:,i,:,:]=seg_cpu==i
    seg_one_hot_tensor=torch.from_numpy(seg_one_hot).to(device)
    return seg_one_hot_tensor

def get_dice_loss_fn(nb_classes=3,epsilon=1e-10,include_bg=False,one_hot=True,device='cpu'):
    '''
    Return dice loss function with given parameters
    '''
    return lambda target, output: dice_loss(target, output, nb_classes=nb_classes,epsilon=epsilon,
                include_bg=include_bg,device=device,one_hot=one_hot)

def dice_loss(output, target, nb_classes=3,epsilon=1e-10,
              include_bg=False,one_hot=True,device='cpu'):
    '''
    Calculate dice loss for given target and output.
    :param output: predicted segmentation
    :param target: target segmentation
    :param nb_classes: number of classes
    :param epsilon: epsilon parameter for numerical stability
    :param include_bg: whether to include background in loss calculation or not
    :param one_hot: whether the target and output are in one-hot encoding or not. If not, they are converted to one-hot
                    encoding.
    :param device: device used by pytorch
    :return: dice loss
    '''
    if not one_hot:
        target=convert_to_one_hot(target,nb_classes=nb_classes,device=device)
        output=convert_to_one_hot(output,nb_classes=nb_classes,device=device)
    smooth = 1.
    dice = 0
    if include_bg:
        start_idx=0
    else:
        start_idx=1
    for obj in range(start_idx, nb_classes):
        output_obj = output[:, obj, :, :]
        target_obj = target[:, obj, :, :]
        intersection_obj = torch.sum(output_obj * target_obj)
        union_obj = torch.sum(output_obj * output_obj) + torch.sum(target_obj * target_obj)
        dice += (2. * intersection_obj + smooth) / (union_obj + smooth)
    dice /= (nb_classes - 1)
    return - torch.clamp(dice, 0., 1. - epsilon)

if __name__ == '__main__':
    # quick test code
    seg = np.array([[0, 0, 1, 1, 0, 0],
                   [0, 0, 1, 1, 0, 0]])
    gt = np.array([[0, 0, 0, 1, 0, 0],
                   [0, 0, 1, 1, 0, 0]])
    print(dice_score(seg, gt,[1]))



