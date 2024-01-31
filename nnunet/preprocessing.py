import nibabel as nib
import os
from tqdm import tqdm
import json
import sys
import CONST
import yaml
import numpy as np
import utils


def create_nnunet_folder_if_not_exist(nnunet_folder_loc):
    '''
    Create nnunet folder structure if it does not exist yet
    :param nnunet_folder_loc: location of nnunet folder
    :return: locations of imagesTr and labelsTr folders
    '''
    if not os.path.exists(nnunet_folder_loc):
        os.makedirs(nnunet_folder_loc)
    images_tr_loc=os.path.join(nnunet_folder_loc,'imagesTr')
    if not os.path.exists(images_tr_loc):
        os.makedirs(images_tr_loc)
    labels_tr_loc=os.path.join(nnunet_folder_loc,'labelsTr')
    if not os.path.exists(labels_tr_loc):
        os.makedirs(labels_tr_loc)
    return images_tr_loc,labels_tr_loc


def convert_to_nnunet_format(config_loc, verbose=True):
    '''
    Convert CAMUS data to nnUNet format with given config file.
    This function will either create a nnunet folder structure in your nnunet_raw_data folder under the dataset id specified
    in the config file or overwrite the existing nnunet folder structure at that location.
    The nnunet folder structure will be:
    |-nnunet_raw_data
    | |-DatasetXXX_CAMUS_trainval
    | | |-imagesTr
    | | |-labelsTr
    | |-DatasetYYY_CAMUS_test
    | | |-imagesTr
    | | |-labelsTr
    Where XXX is the dataset id specified in the config file and YYY is XXX+1. The trainval folder contains the
    training and validation data, and the test folder contains the test data. The imagesTr folder contains the
    ultrasound images, and the labelsTr folder contains the ground truth segmentations.
    Additionally, a splits_final.json file will be created in the splits_out_loc folder specified in the config file.
    This file contains the splits used for training, validation and testing in nnunet format.
    :param config_loc: location of config file
    :param verbose: whether to print info or not
    '''
    if verbose:
        print('Running evaluation with config file: ' + config_loc)
    config = yaml.load(open(config_loc), Loader=yaml.loader.SafeLoader)
    splits=utils.get_splits(config['SPLIT_NB'], config['CAMUS_SPLITS_LOCATION'])
    train_set,val_set,test_set=splits
    nnunet_dataset_id=config['NNUNET_DATASET_ID']
    dataset_id_str=str(nnunet_dataset_id).zfill(3)
    train_val_loc=os.path.join(config['NNUNET_RAW_LOC'],f'Dataset{dataset_id_str}_CAMUS_trainval')
    nnunet_dataset_id+=1
    dataset_id_str=str(nnunet_dataset_id).zfill(3)
    test_loc=os.path.join(config['NNUNET_RAW_LOC'],f'Dataset{dataset_id_str}_CAMUS_test')
    images_tr_loc_trainval,labels_tr_loc_trainval=create_nnunet_folder_if_not_exist(train_val_loc)
    images_tr_loc_test,labels_tr_loc_test=create_nnunet_folder_if_not_exist(test_loc)
    # the nnunet specific splits info for the 'splits_final.json' file
    splits_info = {}
    splits_info['train'] = []
    splits_info['val'] = []
    splits_info['test'] = []

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
                    us_png=utils.resize_image(us_npy)
                    nii_img_gt  = nib.load(file_path)
                    gt_npy = nii_img_gt.get_fdata()
                    gt_png=utils.resize_image(gt_npy,annotation=True)
                    save_name=file_us_img.replace('.nii.gz','')
                    save_name_us=save_name+'_0000.png'
                    save_name_gt=save_name+'.png'
                    if patient in train_set or patient in val_set:
                        # save to trainval folder
                        us_png.save(os.path.join(train_val_loc,images_tr_loc_trainval,save_name_us))
                        gt_png.save(os.path.join(train_val_loc,labels_tr_loc_trainval,save_name_gt))
                    elif patient in test_set:
                        # save to test folder
                        us_png.save(os.path.join(test_loc,images_tr_loc_test,save_name_us))
                        gt_png.save(os.path.join(test_loc,labels_tr_loc_test,save_name_gt))

                    if patient in train_set:
                        splits_info['train'].append(save_name)
                    elif patient in val_set:
                        splits_info['val'].append(save_name)
                    elif patient in test_set:
                        splits_info['test'].append(save_name)

    save_loc_split = os.path.join(config['SPLITS_OUT_LOC'], 'splits_final.json')
    with open(save_loc_split, 'w') as f:
        json.dump([splits_info], f)



if __name__ == '__main__':
    # load config file if provided, otherwise use default
    if len(sys.argv) > 1:
        config_loc = sys.argv[1]
    else:
        config_loc = CONST.DEFAULT_PREPROCESSING_CONFIG_LOC

    print('Converting CAMUS data to nnUNet format..')
    convert_to_nnunet_format(config_loc)


















