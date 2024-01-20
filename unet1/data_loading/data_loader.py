import os
import numpy as np
import torch
#from data_loading import augmentations
import augmentations
import matplotlib.pyplot as plt


class Labeled_dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for unlabeled data of HUNT4'

    def __init__(self, patient_list, input_dir, augmentation_params=None, verbose=True,
                 return_file_loc=False):
        'Initialization'
        self.verbose = verbose
        self.patient_list = patient_list
        self.input_dir = input_dir
        self.init_index_to_file_dict()
        self.return_file_loc = return_file_loc
        if augmentation_params is None:
            self.augmentations = None
        else:
            self.augmentations = augmentations.get_augmentation_funcs(augmentation_params)



    def __len__(self):
        'Denotes the total number of patients'
        return len(self.list_IDs)

    def __getitem__(self, custom_index=None):
        'Generates one sample of data'
        if custom_index is None:
            index = self.index
        else:
            index = custom_index
        # Generate data
        if self.return_file_loc:
            X,y,file_loc = self.__data_generation(index)
        else:
            X,y = self.__data_generation(index)
        y = np.squeeze(y)

        if self.augmentations is not None:
            X,y = augmentations.apply_augmentations((X,y), self.augmentations)

        if isinstance(X, list):
            X = [torch.from_numpy(x.copy()) for x in X]
        else:
            X = torch.from_numpy(X.copy())
        if self.return_file_loc:
            return X, y, file_loc
        else:
            return X,y

    def __data_generation(self, index):
        'Generates data containing batch_size samples' # X : (n_samples, *dim)
        # Generate data
        file_loc= self.index_to_file_dict[index]
        X,y = np.load(file_loc,allow_pickle=True)
        if self.return_file_loc:
            return X,y,file_loc
        else:
            return X, y


    def init_index_to_file_dict(self):
        self.index_to_file_dict = {}
        index=0
        for patient_id in os.listdir(self.input_dir):
            if patient_id in self.patient_list:
                patient_dir = os.path.join(self.input_dir,patient_id)
                for exam in os.listdir(patient_dir):
                    exam_dir = os.path.join(patient_dir,exam)
                    for recording in os.listdir(exam_dir):
                        recording_path = os.path.join(exam_dir,recording)
                        self.index_to_file_dict[index]=(recording_path)
                        index+=1
        if self.verbose:
            print('Number of recordings in dataset:' +str(index))
        self.list_IDs = np.arange(index)

