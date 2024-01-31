import os
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
import utils
from tqdm import tqdm
import yaml
import CONST
import sys


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

def run_eval(config_loc,verbose=True):
    '''
    Run evaluation with given config file.
    This function will create a folder in the experiments directory with the relaatve path specified in the config file
    by the key REL_OUT_DIR. In this folder, a plots folder will be created to save the plots of the prediction and
    annotation for each sample. The plots will be saved as png files with the same name as the sample. Additionally,
    a boxplot of the dice scores will be created and saved as boxplot.png in the same folder.
    :param config_loc: location of config file
    :param verbose: whether to print info or not
    '''
    if verbose:
        print('Running evaluation with config file: ' + config_loc)
    config = yaml.load(open(config_loc), Loader=yaml.loader.SafeLoader)
    out_dir=os.path.join(CONST.EXPERIMENTS_DIR,config['REL_OUT_DIR'])
    if verbose:
        print('Saving results to: ' + out_dir)
    plot_folder=os.path.join(out_dir,'plots')
    if not os.path.exists(CONST.EXPERIMENTS_DIR):
        os.makedirs(CONST.EXPERIMENTS_DIR)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    all_dices=[]

    for sample in tqdm(os.listdir(config['INPUT_DIR_IMAGES'])):
        #repalce _0000.png by .png
        sample_name=sample.replace('_0000.png','.png')
        anno_loc=os.path.join(config['INPUT_DIR_LABELS'],sample_name)
        pred_loc=os.path.join(config['INFERENCE_DIR'],sample_name)
        # transpose images to get correct orientation for plotting
        us_image=imread(os.path.join(config['INPUT_DIR_IMAGES'],sample)).T
        anno=imread(anno_loc).T
        pred=imread(pred_loc).T
        # labels are 1,2,3 for left ventricle, myocardium and left atrium respectively
        dice_lv=utils.dice_score(anno,pred,[1])
        dice_myo=utils.dice_score(anno,pred,[2])
        dice_la=utils.dice_score(anno,pred,[3])
        dices=[dice_lv,dice_myo,dice_la]
        # add dice to list
        all_dices.append(dices)
        plot_segmentation(us_image, anno, pred, sample_name, dices, plot_folder)

    all_dices=np.array(all_dices)
    all_dices_per_class=all_dices.T
    title_dice_scores=('Boxplot of Dice scores per label: \n'
           'Average LV Dice: ') + str(np.round(np.mean(all_dices_per_class[0]),2))+\
            ', Average Myo Dice: ' + str(np.round(np.mean(all_dices_per_class[1]),2))+\
            ', Average LA Dice: ' + str(np.round(np.mean(all_dices_per_class[2]),2))+\
            '\n'
    boxplot(all_dices, out_dir, title_dice_scores,
            'Dice score',['LV','Myo','LA'],'boxplot_dices.png')


if __name__ == '__main__':
    # load config file if provided, otherwise use default
    if len(sys.argv) > 1:
        config_loc = sys.argv[1]
    else:
        config_loc = CONST.DEFAULT_EVAL_CONFIG_LOC

    run_eval(config_loc)

