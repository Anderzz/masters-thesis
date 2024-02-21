import yaml
import sys
import CONST
from datetime import datetime
from data_loading import data_loader
import torch
import os
import numpy as np
import utils
from torch.utils.tensorboard import SummaryWriter
import network
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2


def run_model(dataloader, optimizer, model, loss_fn, train=True, device=None):
    """
    Run the model for one epoch on the given dataloader
    :param dataloader: torch.utils.data.DataLoader to run on
    :param optimizer: torch.optim.Optimizer to use for training
    :param model: torch.nn.Module to run
    :param loss_fn: loss function to use
    :param train: bool, whether to train or not. If False, no gradients will be computed
    :param device: torch.device to run on. Used by pytorch
    :return: average loss over the epoch
    """
    losses = []
    dice_scores = []
    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        labels_one_hot = utils.convert_to_one_hot(labels, device=device)
        # Zero your gradients for every batch!
        optimizer.zero_grad()
        # Make predictions for this batch
        if train:
            model.train()
        else:
            model.eval()
        # add channel dimension to inputs
        # inputs = inputs.unsqueeze(1)
        predictions = model.forward(inputs)
        # Compute the loss and its gradients
        if not train:
            with torch.no_grad():
                batch_loss = loss_fn(predictions, labels_one_hot)
        else:
            batch_loss = loss_fn(predictions, labels_one_hot)
            batch_loss.backward()
        predictions = torch.argmax(predictions, dim=1)
        predictions = predictions.cpu().numpy()
        labels = labels.cpu().numpy().astype(int)
        batch_dice = utils.dice_score(predictions, labels, labels=[0, 1, 2, 3])
        losses.append(batch_loss.item())
        dice_scores.append(batch_dice)
        if train:
            # Adjust learning weights
            optimizer.step()
    return np.mean(losses), np.mean(dice_scores)


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


def train(config_loc, verbose=True):
    """
    Train the model according to the given config file. The function will create a new subdirectory
    in the experiments directory at the location specified in the config file. In this subdirectory,
    a logs folder will be created to save the tensorboard logs. Additionally, each time the validation
    loss is lower than the previous lowest validation loss, the model will be saved in this subdirectory.
    :param config_loc: str, path to the config file
    :param verbose: bool, whether to print progress or not
    """
    if verbose:
        print("Running preprocessing with config file: " + config_loc)
    config = yaml.load(open(config_loc), Loader=yaml.loader.SafeLoader)
    splits = utils.get_splits(config["SPLIT_NB"], config["CAMUS_SPLITS_LOCATION"])
    train_set, val_set, test_set = splits
    train_val_loc = os.path.join(config["PREPROCESSING_OUT_LOC"], "train_val")
    device = utils.set_up_gpu(config["GPU"])
    print(f'Running on device: {device} for {config["TRAINING"]["NB_EPOCHS"]} epochs.')

    # Create a unique directory for this run based on the current timestamp
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(
        CONST.EXPERIMENTS_DIR, config["REL_OUT_DIR"], current_time
    )
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

    # Create a separate logs directory within the unique run directory
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    loss_fn = get_loss(config["TRAINING"]["LOSS"], device)
    input_shape = config["MODEL"]["INPUT_SHAPE"]
    # convert string to tuple
    input_shape_tuple = tuple([int(x) for x in input_shape.split(",")])
    model = network.unet1(input_shape_tuple)
    model = model.to(device)
    # model.double()
    train_transform = A.Compose(
        [
            A.ShiftScaleRotate(
                shift_limit=0.2, scale_limit=0.2, rotate_limit=10, p=0.5
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.25),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.25),
            # A.ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
            # A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            # A.Normalize(mean=(0.485), std=(0.229)),
            ToTensorV2(),
        ]
    )
    val_transform = A.Compose(
        [
            # A.Normalize(mean=(0.485), std=(0.229)),
            ToTensorV2(),
        ]
    )

    # Serialize the augmentations
    train_augmentations_serialized = utils.serialize_augmentations(train_transform)
    # val_augmentations_serialized = utils.serialize_augmentations(val_transform)
    # Save the config file in the unique run directory and dump what augmentations are used
    with open(os.path.join(output_dir, "config.yaml"), "w") as file:
        # add the augmentations to the config file
        config["TRAINING"]["AUGMENTATION_PARAMS"] = train_augmentations_serialized
        yaml.dump(config, file)

    if verbose:
        total_nb_params = sum(p.numel() for p in model.parameters())
        print("total number of params: " + str(total_nb_params))
    dataset_train = data_loader.Labeled_dataset(
        train_set,
        train_val_loc,
        augmentation_params=config["TRAINING"]["AUGMENTATION_PARAMS"],
        transform=train_transform,
    )
    data_loader_params = config["TRAINING"]["DATA_LOADER_PARAMS"]
    dataloader_train = torch.utils.data.DataLoader(dataset_train, **data_loader_params)
    dataset_validation = data_loader.Labeled_dataset(
        val_set, train_val_loc, transform=val_transform
    )
    dataloader_validation = torch.utils.data.DataLoader(
        dataset_validation, **data_loader_params
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config["TRAINING"]["LR"])
    min_loss = np.inf
    writer = SummaryWriter(log_dir=os.path.join(output_dir, "logs"))
    # calculate loss on validation set before first epoch
    validation_loss, validation_dice = run_model(
        dataloader_validation, optimizer, model, loss_fn, train=False, device=device
    )
    writer.add_scalar("Loss/validation", validation_loss, 0)
    writer.add_scalar("Dice/validation", validation_dice, 0)
    print(f"Epoch 0 validation dice: {validation_dice}")
    torch.save(
        model.state_dict(),
        output_dir
        + "/model_epoch_0_dice_"
        + str(np.round(validation_dice, 2))
        + ".pth",
    )
    writer.flush()
    for epoch in range(config["TRAINING"]["NB_EPOCHS"]):
        train_loss, train_dice = run_model(
            dataloader_train, optimizer, model, loss_fn, train=True, device=device
        )
        validation_loss, validation_dice = run_model(
            dataloader_validation, optimizer, model, loss_fn, train=False, device=device
        )
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Dice/train", train_dice, epoch)
        writer.add_scalar("Loss/validation", validation_loss, epoch)
        writer.add_scalar("Dice/validation", validation_dice, epoch)
        if validation_loss < min_loss:
            min_loss = validation_loss
            torch.save(
                model.state_dict(),
                output_dir
                + "/model_epoch_"
                + str(epoch + 1)
                + "_dice_"
                + str(np.round(validation_dice, 2))
                + ".pth",
            )

        print(
            f'Epoch {epoch + 1}/{config["TRAINING"]["NB_EPOCHS"]} validation loss: {validation_loss}'
        )
        print(
            f'Epoch {epoch + 1}/{config["TRAINING"]["NB_EPOCHS"]} validation dice: {validation_dice}'
        )
        writer.flush()
    torch.save(
        model.state_dict(),
        output_dir
        + "/model_epoch_"
        + str(epoch + 1)
        + "_dice_"
        + str(np.round(validation_dice, 2))
        + ".pth",
    )
    writer.close()


if __name__ == "__main__":
    # load config file if provided, otherwise use default
    if len(sys.argv) > 1:
        config_loc = sys.argv[1]
    else:
        config_loc = CONST.DEFAULT_TRAINING_CONFIG_LOC

    print("Converting CAMUS data to numpy format..")
    train(config_loc)
