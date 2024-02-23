import torch
import yaml
import os
import sys
import CONST
import numpy as np
from torch.utils.data import DataLoader
from data_loading import data_loader
import network
import albumentations as A
from albumentations.pytorch import ToTensorV2
import utils
from tqdm import tqdm
import matplotlib.pyplot as plt


def get_output_dir(config):
    """
    Make <plots> directory in the same directory as the model to save the outputs
    :param config: dict, configuration dictionary
    :return: str, output directory
    """
    out_dir = "/".join(config["MODEL"]["PATH_TO_MODEL"].split("/")[:-1])
    plot_folder = os.path.join(out_dir, "plots")
    print("Saving results to: " + plot_folder)
    os.makedirs(plot_folder, exist_ok=True)
    return out_dir


def get_loss(loss_name, device):
    """
    Get the loss function to use for training
    :param loss_name: str, name of the loss function
    :param device: torch.device to run on. Used by pytorch
    :return: loss function
    """
    if loss_name == "DICE":
        loss_fn = utils.get_dice_loss_fn(device=device)
    else:
        raise NotImplementedError
    return loss_fn


def test(config_loc):
    print("Loading configuration from:", config_loc)
    with open(config_loc, "r") as file:
        config = yaml.load(file, Loader=yaml.loader.SafeLoader)

    # out_dir = os.path.join(CONST.EXPERIMENTS_DIR, config["REL_OUT_DIR"])
    out_dir = "/".join(config["MODEL"]["PATH_TO_MODEL"].split("/")[:-1])
    plot_folder = os.path.join(out_dir, "plots")
    print("Saving results to: " + plot_folder)
    os.makedirs(plot_folder, exist_ok=True)

    splits = utils.get_splits(config["SPLIT_NB"], config["CAMUS_SPLITS_LOCATION"])
    train_set, val_set, test_set = splits
    test_loc = os.path.join(config["PREPROCESSING_OUT_LOC"], "test")
    device = utils.set_up_gpu(config["GPU"])

    model_path = config["MODEL"]["PATH_TO_MODEL"]
    input_shape = config["MODEL"]["INPUT_SHAPE"]
    input_shape_tuple = tuple([int(x) for x in input_shape.split(",")])
    model = network.unet1(
        input_shape_tuple, normalize_input=True, normalize_inter_layer=True
    )
    # model = network.unet1(input_shape_tuple)
    # model = network.unet1_transconv(
    #     input_shape_tuple, normalize_input=True, normalize_inter_layer=True
    # )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    loss_fn = get_loss(config["TESTING"]["LOSS"], device)

    val_transform = A.Compose(
        [
            # A.Normalize(mean=(0.485), std=(0.229)),
            ToTensorV2(),
        ]
    )

    data_loader_params = config["TESTING"]["DATA_LOADER_PARAMS"]
    dataset_test = data_loader.Labeled_dataset(
        test_set, test_loc, transform=val_transform
    )
    dataloader_test = torch.utils.data.DataLoader(dataset_test, **data_loader_params)
    losses = []
    dice_scores = []
    print(len(dataset_test))
    with torch.no_grad():
        for i, (inputs, labels) in tqdm(
            enumerate(dataloader_test), total=len(dataloader_test)
        ):
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels_one_hot = utils.convert_to_one_hot(labels, device=device)
            predictions = model(inputs)
            loss = loss_fn(predictions, labels_one_hot).cpu().numpy()
            predictions = torch.argmax(predictions, dim=1)
            predictions = predictions.cpu().numpy()
            labels = labels.cpu().numpy().astype(int)

            # dice = utils.dice_score(predictions, labels, [0, 1, 2, 3])
            dice_lv = utils.dice_score(predictions.squeeze(), labels.squeeze(), [1])
            dice_myo = utils.dice_score(predictions.squeeze(), labels.squeeze(), [2])
            dice_la = utils.dice_score(predictions.squeeze(), labels.squeeze(), [3])
            dices = [dice_lv, dice_myo, dice_la]
            av_dice = np.mean(dices)

            utils.plot_segmentation(
                inputs.cpu().numpy().squeeze().T,
                labels.squeeze().T,
                predictions.squeeze().T,
                f"{i}.png",
                dices,
                plot_folder,
            )
            losses.append(loss)
            # dice_scores.append(dice)
            dice_scores.append(av_dice)

    avg_loss = np.mean(losses)
    avg_dice = np.mean(dice_scores, axis=0)
    # return avg_loss, avg_dice, dices
    return avg_loss, av_dice, dices


def main():
    # load config file if provided, otherwise use default
    if len(sys.argv) > 1:
        config_loc = sys.argv[1]
    else:
        config_loc = CONST.DEFAULT_TESTING_CONFIG_LOC
    loss, dice, dices = test(config_loc)
    print(f"Average loss: {loss}")
    print(f"Average dice score: {dice}")
    print(f"Dice scores: [LV, MYO, LA] = {[round(x, 3) for x in dices]}")


if __name__ == "__main__":
    main()
