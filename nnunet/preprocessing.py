import nibabel as nib
import os
from PIL import Image
from skimage.transform import resize
from tqdm import tqdm
import json


#TODO: clean, structure and document this code


def text_to_set(text_loc):
    with open(text_loc) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    return set(content)

def get_splits(split_nb,splits_loc):
    train_loc = os.path.join(str(splits_loc), f'subGroup{split_nb}_testing.txt')
    val_loc = os.path.join(str(splits_loc), f'subGroup{split_nb}_validation.txt')
    test_loc = os.path.join(str(splits_loc), f'subGroup{split_nb}_training.txt')
    train_set = text_to_set(train_loc)
    val_set = text_to_set(val_loc)
    test_set = text_to_set(test_loc)
    return train_set,val_set,test_set

def convert_to_png(numpy_image,resize_dim=(256,256)):
    numpy_image_resized=resize(numpy_image,resize_dim)
    img_data = Image.fromarray(numpy_image_resized)
    img_data_grayscale = img_data.convert("L")
    return img_data_grayscale

def create_nnunet_folder_if_not_exist(nnunet_folder_loc):
    if not os.path.exists(nnunet_folder_loc):
        os.makedirs(nnunet_folder_loc)
    images_tr_loc=os.path.join(nnunet_folder_loc,'imagesTr')
    if not os.path.exists(images_tr_loc):
        os.makedirs(images_tr_loc)
    labels_tr_loc=os.path.join(nnunet_folder_loc,'labelsTr')
    if not os.path.exists(labels_tr_loc):
        os.makedirs(labels_tr_loc)
    return images_tr_loc,labels_tr_loc

def convert_to_nnunet_format(camus_data_location,nnunet_raw_loc,nnunet_dataset_id,splits,splits_out_loc):
    train_set,val_set,test_set=splits
    dataset_id_str=str(nnunet_dataset_id).zfill(3)
    train_val_loc=os.path.join(nnunet_raw_loc,f'Dataset{dataset_id_str}_CAMUS_trainval')
    nnunet_dataset_id+=1
    dataset_id_str=str(nnunet_dataset_id).zfill(3)
    test_loc=os.path.join(nnunet_raw_loc,f'Dataset{dataset_id_str}_CAMUS_test')
    images_tr_loc_trainval,labels_tr_loc_trainval=create_nnunet_folder_if_not_exist(train_val_loc)
    images_tr_loc_test,labels_tr_loc_test=create_nnunet_folder_if_not_exist(test_loc)
    # the nnunet specific splits info for the 'splits_final.json' file
    splits_info = {}
    splits_info['train'] = []
    splits_info['val'] = []
    splits_info['test'] = []

    for patient in tqdm(os.listdir(camus_data_location)):
        patient_path=os.path.join(camus_data_location,patient)
        if os.path.isdir(patient_path):
            for file in os.listdir(patient_path):
                file_path=os.path.join(patient_path,file)
                # only use ED and ES (end diastole and end systole)
                # gt stands for ground truth
                if file.endswith('.nii.gz') and ('ED' in file or 'ES' in file) and 'gt' in file:
                    file_us_img=file.replace('_gt','')
                    nii_img_us  = nib.load(os.path.join(patient_path,file_us_img))
                    us_npy = nii_img_us.get_fdata()
                    us_png=convert_to_png(us_npy)
                    nii_img_gt  = nib.load(file_path)
                    gt_npy = nii_img_gt.get_fdata()
                    gt_png=convert_to_png(gt_npy)
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

    save_loc_split = os.path.join(splits_out_loc, 'splits_final.json')
    with open(save_loc_split, 'w') as f:
        json.dump([splits_info], f)



if __name__ == '__main__':
    # TODO: put in yaml config file
    camus_data_location = '../local_data/database_nifti'
    camus_splits_location = '../local_data/subgroups_CAMUS'
    nnunet_raw_loc='/home/gillesv/data/nnUNet_raw'
    split_nb = 1
    nnunet_dataset_id=105
    splits=get_splits(split_nb, camus_splits_location)
    splits_out_loc='preprocessing_output'

    print('Converting CAMUS data to nnUNet format..')
    convert_to_nnunet_format(camus_data_location,nnunet_raw_loc,nnunet_dataset_id,splits,splits_out_loc)




















