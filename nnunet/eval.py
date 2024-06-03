from skimage.io import imread
import matplotlib.pyplot as plt
import utils
import os
from tqdm import tqdm
import sys
import CONST
import yaml
import numpy as np


def run_eval(config_loc, verbose=True):
    """
    Run evaluation with given config file.
    This function will create a folder in the experiments directory with the relaatve path specified in the config file
    by the key REL_OUT_DIR. In this folder, a plots folder will be created to save the plots of the prediction and
    annotation for each sample. The plots will be saved as png files with the same name as the sample. Additionally,
    a boxplot of the dice scores will be created and saved as boxplot.png in the same folder.
    :param config_loc: location of config file
    :param verbose: whether to print info or not
    """
    if verbose:
        print("Running evaluation with config file: " + config_loc)
    config = yaml.load(open(config_loc), Loader=yaml.loader.SafeLoader)
    out_dir = os.path.join(CONST.EXPERIMENTS_DIR, config["REL_OUT_DIR"])
    if verbose:
        print("Saving results to: " + out_dir)
    plot_folder = os.path.join(out_dir, "plots")
    if not os.path.exists(CONST.EXPERIMENTS_DIR):
        os.makedirs(CONST.EXPERIMENTS_DIR)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    all_dices = []
    hausdorff_scores = []

    for sample in tqdm(os.listdir(config["INPUT_DIR_IMAGES"])):
        # repalce _0000.png by .png
        sample_name = sample.replace("_0000.png", ".png")
        anno_loc = os.path.join(config["INPUT_DIR_LABELS"], sample_name)
        pred_loc = os.path.join(config["INFERENCE_DIR"], sample_name)
        # transpose images to get correct orientation for plotting
        us_image = imread(os.path.join(config["INPUT_DIR_IMAGES"], sample)).T
        anno = imread(anno_loc).T
        pred = imread(pred_loc).T
        # labels are 1,2,3 for left ventricle, myocardium and left atrium respectively
        dice_lv = utils.dice_score(anno, pred, [1])
        dice_myo = utils.dice_score(anno, pred, [2])
        dice_la = utils.dice_score(anno, pred, [3])
        dices = [dice_lv, dice_myo, dice_la]
        # add dice to list
        all_dices.append(dices)
        utils.plot_segmentation(us_image, anno, pred, sample_name, dices, plot_folder)

        # hausdorf distances
        hausdorf_lv = utils.hausdorf(pred.squeeze(), anno.squeeze(), 1)
        hausdorf_myo = utils.hausdorf(pred.squeeze(), anno.squeeze(), 2)
        hausdorf_la = utils.hausdorf(pred.squeeze(), anno.squeeze(), 3)
        hausdorff_scores.append([hausdorf_lv, hausdorf_myo, hausdorf_la])

    all_dices = np.array(all_dices)
    all_dices_per_class = all_dices.T
    hausdorff_scores = np.array(hausdorff_scores)
    hausdorff_per_class = hausdorff_scores.T
    title_dice_scores = (
        ("Average Dice scores:\n LV: ")
        + str(np.round(np.mean(all_dices_per_class[0]), 2))
        + ", Myo: "
        + str(np.round(np.mean(all_dices_per_class[1]), 2))
        + ", LA: "
        + str(np.round(np.mean(all_dices_per_class[2]), 2))
        + "\n"
    )
    title_hausdorf_scores = (
        ("Average Hausdorff distances:\n LV: ")
        + str(np.round(np.mean(hausdorff_per_class[0]), 2))
        + ", MYO: "
        + str(np.round(np.mean(hausdorff_per_class[1]), 2))
        + ", LA: "
        + str(np.round(np.mean(hausdorff_per_class[2]), 2))
        + "\n"
    )
    utils.boxplot(
        hausdorff_scores,
        out_dir,
        title_hausdorf_scores,
        "Hausdorff distance",
        ["LV", "Myo", "LA"],
        "boxplot_hausdorff.png",
        metric="hausdorff",
    )
    utils.boxplot(
        all_dices,
        out_dir,
        title_dice_scores,
        "Dice score",
        ["LV", "Myo", "LA"],
        "boxplot_dices.png",
        metric="dice",
    )
    print(
        f"Average dice scores: {np.round(np.mean(all_dices), 3)} | {np.round(np.mean(all_dices, axis=0), 3)}"
    )
    raw_scores_file = os.path.join(out_dir, "raw_scores.txt")
    with open(raw_scores_file, "w") as f:
        f.write("Dice Scores:\n")
        for dice in all_dices:
            f.write(", ".join(map(str, dice)) + "\n")

        f.write("\nHausdorff Distances:\n")
        for hausdorf in hausdorff_scores:
            f.write(", ".join(map(str, hausdorf)) + "\n")


if __name__ == "__main__":
    # load config file if provided, otherwise use default
    if len(sys.argv) > 1:
        config_loc = sys.argv[1]
    else:
        config_loc = CONST.DEFAULT_EVAL_CONFIG_LOC

    run_eval(config_loc)
