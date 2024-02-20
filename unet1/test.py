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

    splits = utils.get_splits(config["SPLIT_NB"], config["CAMUS_SPLITS_LOCATION"])
    train_set, val_set, test_set = splits
    test_loc = os.path.join(config["PREPROCESSING_OUT_LOC"], "test")
    device = utils.set_up_gpu(config["GPU"])
    print(f"Running on device: {device}")

    model_path = config["MODEL"]["PATH_TO_MODEL"]
    input_shape = config["MODEL"]["INPUT_SHAPE"]
    input_shape_tuple = tuple([int(x) for x in input_shape.split(",")])
    model = network.unet1(input_shape_tuple)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    loss_fn = get_loss(config["TESTING"]["LOSS"], device)

    val_transform = A.Compose(
        [
            A.Normalize(mean=(0.485), std=(0.229)),
            ToTensorV2(),
        ]
    )

    data_loader_params = config["TESTING"]["DATA_LOADER_PARAMS"]
    dataset_test = data_loader.Labeled_dataset(
        test_set, test_loc, transform=val_transform
    )
    dataloader_test = torch.utils.data.DataLoader(dataset_test, **data_loader_params)
    losses = []
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader_test, total=len(dataloader_test)):
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels_one_hot = utils.convert_to_one_hot(labels, device=device)
            predictions = model(inputs)
            loss = loss_fn(predictions, labels_one_hot).cpu().numpy()
            losses.append(loss)

    avg_loss = np.mean(losses)
    return avg_loss


def main():
    # load config file if provided, otherwise use default
    if len(sys.argv) > 1:
        config_loc = sys.argv[1]
    else:
        config_loc = CONST.DEFAULT_TESTING_CONFIG_LOC
    loss = test(config_loc)
    print(f"Average loss: {loss}")


if __name__ == "__main__":
    main()
