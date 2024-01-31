import nibabel as nib
import os
from tqdm import tqdm
import yaml
import numpy as np
import utils
import sys
import CONST

def convert_dataset_to_numpy(config_loc, verbose=True):
    '''
    Convert CAMUS data to numpy format with given config file.
    This function will create a folder structure in your preprocessing_out folder under the dataset id specified
    in the config file.
    The folder structure will be:
    |-preprocessing_out
    | |-train_val
    | | |-recording1.npy
    | | |-recording2.npy
    | | | ...
    | |-test
    | | |-recording3.npy
    | | |-recording4.npy
    | | | ...
    Where train_val contains the training and validation data, and test contains the test data.
    Each recording file contains a tuple of two numpy arrays: the ultrasound image and the ground truth segmentation.
    :param config_loc: location of config file
    :param verbose: whether to print info or not
    '''
    if verbose:
        print('Running preprocessing with config file: ' + config_loc)
    config = yaml.load(open(config_loc), Loader=yaml.loader.SafeLoader)
    splits=utils.get_splits(config['SPLIT_NB'], config['CAMUS_SPLITS_LOCATION'])
    train_set,val_set,test_set=splits
    train_val_loc=os.path.join(config['PREPROCESSING_OUT_LOC'],'train_val')
    test_loc=os.path.join(config['PREPROCESSING_OUT_LOC'],'test')
    if not os.path.exists(config['PREPROCESSING_OUT_LOC']):
        os.makedirs(config['PREPROCESSING_OUT_LOC'])
    if not os.path.exists(train_val_loc):
        os.makedirs(train_val_loc)
    if not os.path.exists(test_loc):
        os.makedirs(test_loc)
    for patient in tqdm(os.listdir(config['CAMUS_DATA_LOCATION'])):
        patient_path=os.path.join(config['CAMUS_DATA_LOCATION'],patient)
        if os.path.isdir(patient_path):
            for file in os.listdir(patient_path):
                file_path=os.path.join(patient_path,file)
                # only use ED and ES (end diastole and end systole)
                # gt stands for ground truth
                if file.endswith('.nii.gz') and ('ED' in file or 'ES' in file) and 'gt' in file:
                    file_us_img=file.replace('_gt','')
                    nii_img_us  = nib.load(os.path.join(patient_path,file_us_img))
                    us_npy = nii_img_us.get_fdata()
                    us_resized=utils.resize_image(us_npy,convert_to_png=False)
                    nii_img_gt  = nib.load(file_path)
                    gt_npy = nii_img_gt.get_fdata()
                    gt_resized=utils.resize_image(gt_npy,convert_to_png=False,annotation=True)
                    save_name=file_us_img.replace('.nii.gz','')
                    img_gt_tuple = (us_resized, gt_resized)
                    if patient in train_set or patient in val_set:
                        # save to trainval folder in patient subfolder
                        patient_folder = os.path.join(train_val_loc, patient)
                        if not os.path.exists(patient_folder):
                            os.makedirs(patient_folder)
                        np.save(os.path.join(patient_folder,save_name),img_gt_tuple)
                    elif patient in test_set:
                        # save to test folder in patient subfolder
                        patient_folder = os.path.join(test_loc, patient)
                        if not os.path.exists(patient_folder):
                            os.makedirs(patient_folder)
                        np.save(os.path.join(patient_folder,save_name),img_gt_tuple)

if __name__ == '__main__':
    # load config file if provided, otherwise use default
    if len(sys.argv) > 1:
        config_loc = sys.argv[1]
    else:
        config_loc = CONST.DEFAULT_PREPROCESSING_CONFIG_LOC

    print('Converting CAMUS data to numpy format..')
    convert_dataset_to_numpy(config_loc)