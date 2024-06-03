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
import queue
from scipy.ndimage import label, binary_dilation, binary_erosion


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
        loss_fn = utils.get_dice_deep_supervision_loss_fn(
            device=device, one_hot=True, nb_classes=4, include_bg=False
        )

    else:
        raise NotImplementedError
    return loss_fn


def handle_queue(worst_queue, item, maxsize=15):
    """
    Handle the updates to the priority queue.
    :param worst_queue: queue.PriorityQueue, the priority queue to update
    :param item: tuple, the item to insert into the queue (loss, data)
    :param maxsize: int, the maximum size of the queue
    """
    if worst_queue.qsize() < maxsize:
        worst_queue.put(item)
    else:
        # If the new item has a worse (higher) loss, add it to the queue
        max_item = worst_queue.get()
        if item[0] > max_item[0]:
            worst_queue.put(item)
        else:
            worst_queue.put(max_item)


def keep_largest_component(segmentation):
    # Assuming segmentation is a 2D numpy array with integer class labels
    output_segmentation = np.zeros_like(segmentation)

    # Get unique class labels, ignoring the background (assuming it's labeled as 0)
    class_labels = np.unique(segmentation)[1:]  # Skip 0 if it's the background label

    for class_label in class_labels:
        # Create a binary mask for the current class
        class_mask = segmentation == class_label

        # Perform connected component labeling
        labeled_array, num_features = label(class_mask)

        # Skip if no features are found for the class
        if num_features == 0:
            continue

        # Find the largest component
        largest_component_size = 0
        largest_component_label = 0
        for i in range(1, num_features + 1):  # Component labels are 1-indexed
            component_size = np.sum(labeled_array == i)
            if component_size > largest_component_size:
                largest_component_size = component_size
                largest_component_label = i

        # Keep only the largest component for this class in the output segmentation
        output_segmentation[labeled_array == largest_component_label] = class_label

    return output_segmentation


def opening(segmentation, structure=None):
    if segmentation.ndim == 3 and segmentation.shape[0] == 1:
        segmentation_2d = segmentation[0]  # Access the single 2D image
    else:
        raise ValueError("Segmentation shape is not as expected.")

    # Get unique class labels, ignoring the background (assuming it's labeled as 0)
    class_labels = np.unique(segmentation_2d)
    if 0 in class_labels:
        class_labels = class_labels[1:]  # Skip the background

    # Create a 3x3 square structuring element
    structure = np.ones((3, 3), dtype=bool)

    # Initialize output segmentation array
    output_segmentation = np.zeros_like(segmentation_2d)

    for class_label in class_labels:
        # Create a binary mask for the current class
        class_mask = segmentation_2d == class_label

        # Apply dilation followed by erosion
        eroded_mask = binary_erosion(class_mask, structure=structure)
        dilated_mask = binary_dilation(eroded_mask, structure=structure)

        # Update the output segmentation with the processed class mask
        output_segmentation[dilated_mask] = class_label

    # If needed to fit original shape
    return output_segmentation[
        np.newaxis, :, :
    ]  # Adds the first dimension back if needed


def closing(segmentation, structure=None):
    if segmentation.ndim == 3 and segmentation.shape[0] == 1:
        segmentation_2d = segmentation[0]  # Access the single 2D image
    else:
        raise ValueError("Segmentation shape is not as expected.")

    # Get unique class labels, ignoring the background (assuming it's labeled as 0)
    class_labels = np.unique(segmentation_2d)
    if 0 in class_labels:
        class_labels = class_labels[1:]  # Skip the background

    # Create a 3x3 square structuring element
    structure = np.ones((3, 3), dtype=bool)

    # Initialize output segmentation array
    output_segmentation = np.zeros_like(segmentation_2d)

    for class_label in class_labels:
        # Create a binary mask for the current class
        class_mask = segmentation_2d == class_label

        # Apply dilation followed by erosion
        dilated_mask = binary_dilation(class_mask, structure=structure)
        eroded_mask = binary_erosion(dilated_mask, structure=structure)

        # Update the output segmentation with the processed class mask
        output_segmentation[eroded_mask] = class_label

    # If needed to fit original shape
    return output_segmentation[
        np.newaxis, :, :
    ]  # Adds the first dimension back if needed


def test(config_loc):
    print("Loading configuration from:", config_loc)
    with open(config_loc, "r") as file:
        config = yaml.load(file, Loader=yaml.loader.SafeLoader)

    # out_dir = os.path.join(CONST.EXPERIMENTS_DIR, config["REL_OUT_DIR"])
    out_dir = "/".join(config["MODEL"]["PATH_TO_MODEL"].split("/")[:-1])
    plot_folder = os.path.join(out_dir, "plots")
    plot_folder_ds = os.path.join(out_dir, "plots_ds")
    print("Saving results to: " + plot_folder)
    os.makedirs(plot_folder, exist_ok=True)
    os.makedirs(plot_folder_ds, exist_ok=True)

    splits = utils.get_splits(config["SPLIT_NB"], config["CAMUS_SPLITS_LOCATION"])
    train_set, val_set, test_set = splits
    test_loc = os.path.join(config["PREPROCESSING_OUT_LOC"], "test")
    device = utils.set_up_gpu(config["GPU"])

    model_path = config["MODEL"]["PATH_TO_MODEL"]
    input_shape = config["MODEL"]["INPUT_SHAPE"]
    loss_fn = get_loss(config["TESTING"]["LOSS"], device)
    input_shape_tuple = tuple([int(x) for x in input_shape.split(",")])

    use_ds = config["MODEL"]["DEEP_SUPERVISION"]
    # use_ds = False
    # if use_ds:
    #     loss_fn = get_loss("DICE_DS", device)

    # model = network.unet1(
    #     input_shape_tuple,
    #     activation_inter_layer="mish",
    #     normalize_input=True,
    #     normalize_inter_layer=True,
    #     use_deep_supervision=use_ds,
    # )
    model = network.unet1(
        input_shape_tuple,
        activation_inter_layer="relu",
        normalize_input=False,
        normalize_inter_layer=False,
        use_deep_supervision=use_ds,
    )
    # model = network.unet1(input_shape_tuple)
    # model = network.unet1_transconv(
    #     input_shape_tuple, normalize_input=True, normalize_inter_layer=True
    # )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    total_nb_params = sum(p.numel() for p in model.parameters())
    print("total number of params: " + str(total_nb_params))

    val_transform = A.Compose(
        [
            # A.Normalize(mean=(0.485), std=(0.229)),
            # A.Normalize(mean=(48.6671), std=(53.9987), max_pixel_value=1.0),
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
    hausdorf_scores = []

    # priority queue to store the worst predictions
    worst_queue = queue.PriorityQueue(maxsize=15)

    print(len(dataset_test))
    with torch.no_grad():
        for i, (inputs, labels) in tqdm(
            enumerate(dataloader_test), total=len(dataloader_test)
        ):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # n*h*w x n*h*w
            labels_one_hot = utils.convert_to_one_hot(labels, device=device)
            # inputs = inputs.unsqueeze(1)  # add channel dimension, but ToTensorV2 already does this'
            if use_ds:
                # predictions, *_ = model(inputs)
                predictions, ds1, ds2, ds3, ds4 = model(inputs)
                ds1 = torch.argmax(ds1, dim=1).cpu().numpy()
                ds2 = torch.argmax(ds2, dim=1).cpu().numpy()
                ds3 = torch.argmax(ds3, dim=1).cpu().numpy()
                ds4 = torch.argmax(ds4, dim=1).cpu().numpy()
            else:
                predictions = model(inputs)

            loss = loss_fn(predictions, labels_one_hot).cpu().numpy()
            predictions = torch.argmax(predictions, dim=1)
            predictions = predictions.cpu().numpy()
            labels = labels.cpu().numpy().astype(int)

            # only keep the largest connected component for each class
            # predictions = keep_largest_component(predictions)
            # predictions = opening(predictions)
            predictions = closing(predictions)
            # print(predictions.shape)
            # print(np.unique(predictions, return_counts=True))
            # break

            # dice scores
            dice_lv = utils.dice_score(predictions.squeeze(), labels.squeeze(), [1])
            dice_myo = utils.dice_score(predictions.squeeze(), labels.squeeze(), [2])
            dice_la = utils.dice_score(predictions.squeeze(), labels.squeeze(), [3])
            dices = [dice_lv, dice_myo, dice_la]

            # hausdorf distances
            try:
                hausdorf_lv = utils.hausdorf(predictions.squeeze(), labels.squeeze(), 1)
                hausdorf_myo = utils.hausdorf(
                    predictions.squeeze(), labels.squeeze(), 2
                )
                hausdorf_la = utils.hausdorf(predictions.squeeze(), labels.squeeze(), 3)
                hausdorf_scores.append([hausdorf_lv, hausdorf_myo, hausdorf_la])
            except:
                print("Error in hausdorf distance calculation")
                # hausdorf_lv = 0
                # hausdorf_myo = 0
                # hausdorf_la = 0

            utils.plot_segmentation(
                inputs[0].cpu().numpy().squeeze().T,
                labels[0].squeeze().T,
                predictions[0].squeeze().T,
                f"{i}.png",
                dices,
                plot_folder,
            )

            # # code to plot the upsampled deep supervision outputs
            # if use_ds:
            #     utils.plot_ds_segmentation(
            #         inputs[0].cpu().numpy().squeeze().T,
            #         labels[0].squeeze().T,
            #         ds1[0].squeeze().T,
            #         f"{i}_ds1.png",
            #         dices,
            #         plot_folder_ds,
            #     )
            #     utils.plot_ds_segmentation(
            #         inputs[0].cpu().numpy().squeeze().T,
            #         labels[0].squeeze().T,
            #         ds2[0].squeeze().T,
            #         f"{i}_ds2.png",
            #         dices,
            #         plot_folder_ds,
            #     )
            #     utils.plot_ds_segmentation(
            #         inputs[0].cpu().numpy().squeeze().T,
            #         labels[0].squeeze().T,
            #         ds3[0].squeeze().T,
            #         f"{i}_ds3.png",
            #         dices,
            #         plot_folder_ds,
            #     )
            #     utils.plot_ds_segmentation(
            #         inputs[0].cpu().numpy().squeeze().T,
            #         labels[0].squeeze().T,
            #         ds4[0].squeeze().T,
            #         f"{i}_ds4.png",
            #         dices,
            #         plot_folder_ds,
            #     )

            losses.append(loss)
            dice_scores.append(dices)

            handle_queue(
                worst_queue,
                (loss, (inputs[0].cpu().numpy(), labels[0], predictions[0], dices, i)),
            )
    dice_scores = np.array(dice_scores)
    dice_per_class = dice_scores.T
    hausdorf_scores = np.array(hausdorf_scores)
    hausdorf_per_class = hausdorf_scores.T

    title_dice_scores = (
        ("Average Dice scores:\n LV: ")
        + str(np.round(np.mean(dice_per_class[0]), 2))
        + ", Myo: "
        + str(np.round(np.mean(dice_per_class[1]), 2))
        + ", LA: "
        + str(np.round(np.mean(dice_per_class[2]), 2))
        + "\n"
    )

    title_hausdorf_scores = (
        ("Average Hausdorff distances:\n LV: ")
        + str(np.round(np.mean(hausdorf_per_class[0]), 2))
        + ", MYO: "
        + str(np.round(np.mean(hausdorf_per_class[1]), 2))
        + ", LA: "
        + str(np.round(np.mean(hausdorf_per_class[2]), 2))
        + "\n"
    )
    utils.boxplot(
        hausdorf_scores,
        out_dir,
        title_hausdorf_scores,
        "Hausdorff distance",
        ["LV", "Myo", "LA"],
        "boxplot_hausdorff.png",
        metric="hausdorff",
    )
    utils.boxplot(
        dice_scores,
        out_dir,
        title_dice_scores,
        "Dice score",
        ["LV", "Myo", "LA"],
        "boxplot_dices.png",
        metric="dice",
    )

    raw_scores_file = os.path.join(out_dir, "raw_scores.txt")
    with open(raw_scores_file, "w") as f:
        f.write("Dice Scores:\n")
        for dice in dice_scores:
            f.write(", ".join(map(str, dice)) + "\n")

        f.write("\nHausdorff Distances:\n")
        for hausdorf in hausdorf_scores:
            f.write(", ".join(map(str, hausdorf)) + "\n")

    avg_loss = np.mean(losses)
    avg_dice = np.mean(dice_scores)  # avg across all classes
    avg_dice_per_class = np.mean(
        dice_scores, axis=0
    )  # avg across all samples, but one value per class

    # save the worst predictions
    utils.plot_worst_predictions(worst_queue, plot_folder)

    return avg_loss, avg_dice, avg_dice_per_class


def main():
    # load config file if provided, otherwise use default
    if len(sys.argv) > 1:
        config_loc = sys.argv[1]
    else:
        config_loc = CONST.DEFAULT_TESTING_CONFIG_LOC
    loss, dice, dice_per_class = test(config_loc)
    print(f"Average loss: {round(loss, 3)}")
    print(f"Average dice score: {round(dice, 3)}")
    print(f"Dice scores: [LV, MYO, LA] = {[round(x, 3) for x in dice_per_class]}")
    # save it to a file


if __name__ == "__main__":
    main()
