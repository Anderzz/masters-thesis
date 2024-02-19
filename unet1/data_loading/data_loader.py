import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import utils


class Labeled_dataset(torch.utils.data.Dataset):
    "Characterizes a dataset for labeled segmentation dataset"

    def __init__(
        self,
        patient_list,
        input_dir,
        augmentation_params=None,
        verbose=True,
        return_file_loc=False,
    ):
        """
        :param patient_list: list or set of patient ids to include in dataset
        :param input_dir: directory where data is stored. This directory should contain a folder for each patient,
                            which in turn contains one file for each recording. Each file should be a .npy file
                            containing a tuple of the ultrasound image and its segmentation mask as a numpy arrays.
        :param augmentation_params: list of augmentation parameters
        :param verbose: whether to print information about dataset
        :param return_file_loc: whether to return file location of sample when loading data
        """
        self.verbose = verbose
        self.patient_list = patient_list
        self.input_dir = input_dir
        self.init_index_to_file_dict()
        self.return_file_loc = return_file_loc
        if augmentation_params is None:
            self.augmentations = None
        else:
            self.augmentations = augmentations.get_augmentation_funcs(
                augmentation_params
            )

    def __len__(self):
        "Denotes the total number of patients"
        return len(self.list_IDs)

    def __getitem__(self, custom_index=None):
        """
        'Generates one sample of data'
        :param custom_index: index of sample to load. If None, a random sample will be loaded
        :return: X,y, the sample (ultrasound image) and its label (segmentation mask) as 2D numpy arrays
        """
        if custom_index is None:
            index = self.index
        else:
            index = custom_index
        # Generate data
        if self.return_file_loc:
            X, y, file_loc = self.__data_generation(index)
        else:
            X, y = self.__data_generation(index)
        y = np.squeeze(y)

        if self.augmentations is not None:
            X, y = augmentations.apply_augmentations((X, y), self.augmentations)

        if isinstance(X, list):
            X = [torch.from_numpy(x) for x in X]
        else:
            X = torch.from_numpy(X)
        if self.return_file_loc:
            return X, y, file_loc
        else:
            return X, y

    def __data_generation(self, index):
        "Generates data containing batch_size samples"  # X : (n_samples, *dim)
        # Generate data
        file_loc = self.index_to_file_dict[index]
        X, y = np.load(file_loc, allow_pickle=True)
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        if self.return_file_loc:
            return X, y, file_loc
        else:
            return X, y

    def init_index_to_file_dict(self):
        """
        Initializes a dictionary mapping index to file location. The index is used by the pytorch dataloader to
        load samples. The file location is used to retrieve the file location of a sample when loading data.
        The function sets the following attributes:
            - self.index_to_file_dict: dictionary mapping index to file location
            - self.list_IDs: list of indices
        """
        self.index_to_file_dict = {}
        index = 0
        for patient_id in os.listdir(self.input_dir):
            if patient_id in self.patient_list:
                patient_dir = os.path.join(self.input_dir, patient_id)
                for recording in os.listdir(patient_dir):
                    recording_path = os.path.join(patient_dir, recording)
                    self.index_to_file_dict[index] = recording_path
                    index += 1
        if self.verbose:
            print("Number of recordings in dataset:" + str(index))
        self.list_IDs = np.arange(index)


if __name__ == "__main__":
    import augmentations

    # example usage
    input_dir = "preprocessing_output/cv1/train_val"
    splits = utils.get_splits(1, "../local_data/subgroups_CAMUS")
    train_set, val_set, test_set = splits
    input_dir = "preprocessing_output/cv1/train_val"
    params = {"batch_size": 2, "shuffle": True, "num_workers": 1}
    augmentation_params = []
    dataset_train_pytorch = Labeled_dataset(
        train_set, input_dir, augmentation_params=augmentation_params, verbose=True
    )
    dataloader = torch.utils.data.DataLoader(dataset_train_pytorch, **params)
    train_iterator_pytorch = iter(dataloader)
    # plotting some samples
    for i in range(3):
        X, y = next(train_iterator_pytorch)
        plt.imshow(X[0].T)
        plt.savefig("test.png")
        plt.show()
        plt.imshow(y[0].T)
        plt.show()
        visualization = utils.create_visualization(X[0].T, y[0].T)
        plt.imshow(visualization)
        plt.show()
        plt.clf()
else:
    from data_loading import augmentations
