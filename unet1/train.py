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
from custom_augmentations import *


def save_config(config, augmentations, output_dir):
    """
    Save the config file in the output directory
    :param config: dict, configuration dictionary
    :param output_dir: str, directory to save the config file in
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    # Define the path for the YAML file within the output directory
    config_yaml_path = os.path.join(output_dir, "config.yaml")

    try:
        # Open the file in write mode and save the configuration
        with open(config_yaml_path, "w") as file:
            # Update the configuration dictionary with the serialized augmentations
            config["TRAINING"]["AUGMENTATION_PARAMS"] = augmentations
            # Dump the updated configuration to the YAML file
            yaml.dump(config, file)
        print(f"Configuration saved successfully to {config_yaml_path}")
    except Exception as e:
        print(f"Error saving configuration to YAML: {e}")


def run_model(dataloader, optimizer, model, loss_fn, train=True, device=None, ds=False):
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
    dice_scores = []  # each lsit contains the dice scores for each class per batch
    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        labels_one_hot = utils.convert_to_one_hot(labels, device=device)
        # inputs = inputs.unsqueeze(1)  # add channel dimension, but ToTensorV2 does this for us
        # Zero your gradients for every batch!
        optimizer.zero_grad()
        # Make predictions for this batch
        if train:
            model.train()
        else:
            model.eval()

        if ds:
            predictions, ds1, ds2, ds3, ds4 = model.forward(inputs)
            # print(predictions.shape, ds1.shape, ds2.shape, ds3.shape, ds4.shape)
            # break
        else:
            predictions = model.forward(inputs)  # ["out"]

        # Compute the loss and its gradients
        if not train:
            with torch.no_grad():
                if ds:
                    batch_loss = loss_fn(
                        (predictions, ds1, ds2, ds3, ds4),
                        labels_one_hot,
                    )
                else:
                    batch_loss = loss_fn(predictions, labels_one_hot)
        else:
            if ds:
                batch_loss = loss_fn(
                    (predictions, ds1, ds2, ds3, ds4),
                    labels_one_hot,
                )
            else:
                batch_loss = loss_fn(predictions, labels_one_hot)
            batch_loss.backward()
        # find the class with the highest probability
        predictions = torch.argmax(predictions, dim=1)
        predictions = predictions.cpu().numpy()
        labels = labels.cpu().numpy().astype(int)

        dice_lv = utils.dice_score(predictions.squeeze(), labels.squeeze(), [1])
        dice_myo = utils.dice_score(predictions.squeeze(), labels.squeeze(), [2])
        dice_la = utils.dice_score(predictions.squeeze(), labels.squeeze(), [3])
        dices = [dice_lv, dice_myo, dice_la]
        losses.append(batch_loss.item())
        dice_scores.append(dices)
        if train:
            # Adjust learning weights
            optimizer.step()
    # return avg loss, avg dice score and avg dice score per class
    return np.mean(losses), np.mean(dice_scores), np.mean(dice_scores, axis=0)


def get_loss(loss_name, device):
    """
    Get the loss function to use for training
    :param loss_name: str, name of the loss function
    :param device: torch.device to run on. Used by pytorch
    :return: loss function
    """
    if loss_name == "DICE":
        loss_fn = utils.get_dice_loss_fn(
            device=device, one_hot=True, nb_classes=4, include_bg=False
        )
    elif loss_name == "DICE_WEIGHTED":
        loss_fn = utils.get_weighted_dice_loss_fn(
            class_weights=[1, 1, 1], device=device
        )
    elif loss_name == "DICE&CE":
        print(f"Using DICE&CE loss function")
        loss_fn = utils.get_dice_ce_loss_fn(
            device=device, one_hot=True, nb_classes=4, include_bg=False
        )
    elif loss_name == "DICE_DS":
        print(f"Using DICE_DS loss function")
        loss_fn = utils.get_dice_deep_supervision_loss_fn(
            device=device, one_hot=True, nb_classes=4, include_bg=False
        )

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

    # setup logfile
    log_file_path = os.path.join(output_dir, "train_log.txt")
    sys.stdout = utils.DualLogger(log_file_path)

    # Create a separate logs directory within the unique run directory
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    loss_fn = get_loss(config["TRAINING"]["LOSS"], device)
    input_shape = config["MODEL"]["INPUT_SHAPE"]
    use_ds = config["MODEL"]["DEEP_SUPERVISION"]
    if use_ds:
        loss_fn = get_loss("DICE_DS", device)
    # convert string to tuple
    input_shape_tuple = tuple([int(x) for x in input_shape.split(",")])
    model = network.unet1(
        input_shape_tuple,
        activation_inter_layer="mish",
        normalize_input=True,
        normalize_inter_layer=True,
        use_deep_supervision=use_ds,
    )
    # model = network.getDeeplabv3()
    # model = network.unet1_res(
    #     input_shape_tuple, normalize_input=True, normalize_inter_layer=True
    # )
    # model = network.unet1_transconv(
    #     input_shape_tuple, normalize_input=True, normalize_inter_layer=True
    # )
    model = model.to(device)
    train_transform = A.Compose(
        [
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=(-0.2, 0.1), rotate_limit=10, p=0.5
            ),
            # A.RandomGamma(gamma_limit=(85, 115), p=0.5),
            # A.GaussNoise(var_limit=(10.0, 25.0), p=0.2),
            # Blackout(p=0.25),
            # A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.5),
            ToTensorV2(),
        ]
    )
    # train_transform = A.Compose(
    #     # nnunet transforms
    #     # https://static-content.springer.com/esm/art%3A10.1038%2Fs41592-020-01008-z/MediaObjects/41592_2020_1008_MOESM1_ESM.pdf
    #     [
    #         A.ShiftScaleRotate(
    #             shift_limit=0.0,
    #             scale_limit=(-0.3, 0.4),
    #             rotate_limit=(-180, 180),
    #             p=0.2,
    #         ),
    #         A.GaussNoise((0, 0.1), p=0.15),
    #         A.GaussianBlur(p=0.2),
    #         A.RandomBrightnessContrast(
    #             brightness_limit=(0.7, 1.3), contrast_limit=(0.65, 1.5), p=0.15
    #         ),
    #         A.Downscale(scale_min=0.5, scale_max=0.99, p=0.25),
    #         A.RandomGamma(gamma_limit=(85, 125), p=0.15),
    #         A.Flip(p=0.5),
    #         ToTensorV2(),
    #     ]
    # )
    val_transform = A.Compose(
        [
            # A.Normalize(mean=(0.485), std=(0.229)),
            # A.Normalize(mean=(48.6671), std=(53.9987), max_pixel_value=1.0),
            ToTensorV2(),
        ]
    )

    # Serialize the augmentations
    train_augmentations_serialized = utils.serialize_augmentations(train_transform)

    # save the config to the output directory
    save_config(config, train_augmentations_serialized, output_dir)

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
    print(dataset_train[0][0].shape, dataset_train[0][1].shape)
    dataloader_validation = torch.utils.data.DataLoader(
        dataset_validation, **data_loader_params
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config["TRAINING"]["LR"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", factor=0.1, patience=5
    )
    min_loss = np.inf
    current_best_dice = 0
    writer = SummaryWriter(log_dir=os.path.join(output_dir, "logs"))
    # calculate loss on validation set before first epoch
    validation_loss, validation_dice, validation_dice_per_class = run_model(
        dataloader_validation,
        optimizer,
        model,
        loss_fn,
        train=False,
        device=device,
        ds=use_ds,
    )
    writer.add_scalar("Loss/validation", validation_loss, 0)
    writer.add_scalar("Dice/validation", validation_dice, 0)
    print(
        f"Epoch 0 validation dice: {round(validation_dice, 3)}, per class: {[round(x, 3) for x in validation_dice_per_class]}"
    )
    torch.save(
        model.state_dict(),
        output_dir
        + "/model_epoch_0_dice_"
        + str(np.round(validation_dice, 2))
        + ".pth",
    )
    writer.flush()

    # Early stopping setup
    best_validation_dice = 0
    epochs_without_improvement = 0
    patience = config["TRAINING"]["PATIENCE"]

    for epoch in range(config["TRAINING"]["NB_EPOCHS"]):
        train_loss, train_dice, train_dice_per_class = run_model(
            dataloader_train,
            optimizer,
            model,
            loss_fn,
            train=True,
            device=device,
            ds=use_ds,
        )
        validation_loss, validation_dice, validation_dice_per_class = run_model(
            dataloader_validation,
            optimizer,
            model,
            loss_fn,
            train=False,
            device=device,
            ds=use_ds,
        )

        scheduler.step(validation_dice)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Dice/train", train_dice, epoch)
        writer.add_scalar("Loss/validation", validation_loss, epoch)
        writer.add_scalar("Dice/validation", validation_dice, epoch)

        # Early stopping
        if validation_dice > best_validation_dice:
            best_validation_dice = validation_dice
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement > patience:
            print(
                f"Early stopping after {patience} epochs without improvement. Best validation dice: {best_validation_dice}"
            )
            break

        if validation_dice > current_best_dice:
            current_best_dice = validation_dice
            print(f"New best model, saving..")
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
            f'Epoch {epoch + 1}/{config["TRAINING"]["NB_EPOCHS"]} train loss: {round(train_loss,3)} | val loss: {round(validation_loss,3)}'
        )
        print(
            f'Epoch {epoch + 1}/{config["TRAINING"]["NB_EPOCHS"]} train dice: {round(train_dice,3)} | val dice: {round(validation_dice,3)} {[round(x, 3) for x in validation_dice_per_class]}'
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
    sys.stdout.close()
    sys.stdout = sys.__stdout__  # Restore original stdout


if __name__ == "__main__":
    # load config file if provided, otherwise use default
    if len(sys.argv) > 1:
        config_loc = sys.argv[1]
    else:
        config_loc = CONST.DEFAULT_TRAINING_CONFIG_LOC

    print("Converting CAMUS data to numpy format..")
    train(config_loc)
