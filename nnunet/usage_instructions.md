# Getting started

In this file necessary steps are described to preprocess the data, train the nnU-Net model and evaluate the results.
For this project, you will need nnU-net 


## Set up environment

Create a (conda) envrionment and install the following packages:
- nibabel
- Pillow
- scikit-image
- tqdm
- json
- nnunetv2

TODO: put this in requirements.txt
## Set up nnU-Net
The main nnU-Net github page is [https://github.com/MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet).

Follow the installation instructions on the nnU-Net github page:
[https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md)
If you installed the requirements.txt file in the previous step, you should already have the default nnunet installed.
At a later stage in the project, we will need to modify the nnU-Net code. For this, you will need to clone the nnU-Net
repository and install or run it from source.

Later, we will refer to the nnU-Net environment variables ``` nnUNet_raw```, ``` nnUNet_preprocessed ``` 
and ``` nnUNet_results ```,
described in
[https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/setting_up_paths.md](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/setting_up_paths.md)


## Preprocessing
See the file preprocessing.py.

You should modify the default parameters:
- camus_data_location = location where you donwloaded the camus data.
- camus_splits_location = location of the splits of camus. You don't need to change this.
- nnunet_raw_loc = location of your nnunet raw directory. See above.
- nnunet_dataset_id = id of the nnunet dataset you want to create. See nnunet documentation for more info.
- splits_out_loc = location where you want to save the splits. You don't need to change this.

After changing these parameters, run the preprocessing script.
```bash
python preprocessing.py
```

NOTE: make sure you run in the correct environment (see above).

The preproccesing script will create the imagesTr and labelsTr in the nnU-Net raw directory under the
dataset name and id specified in the preprocessing config file. 

NOTE: We do not rely on nnU-Net to create cross validation 
splits. Instead, we create the splits ourselves and save each split as a separate nnU-Net dataset with only a single
split. This trick gives us more reproducibility, ensures that we split the data on the patient level, and we 
disable ensembling as post-processing step. You should not do ensembling because the final goal is to have a model
that can run in real time and ensembling multiple models is too expensive for this.

If preprocessing was run succesfully, your nnUNet_raw directory should contain 2 new dataset folders, 
one for training and validation ('trainval' )and one for testing ('test'). 
We use the trainval split for training and the test split for inference. It will also create a ```splits_final.json ```
file in the splits_out_loc directory. By default this will be ``` nnunet/preprocessing_output/splits_final.json ```.

I would suggest just using the first cross validation split for now. 
You can always do full cross validation at the end of the project.


## Train nnU-Net

For a full explanation on how to use and configure nnU-Net, see
[https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md).

The instructions below are for training nnU-Net with the default parameters, but using custom cross validation splits that
we generated ourselves in the preprocessing step. We also disable ensembling this way because we save each cross
validation split as a seperate nnU-Net dataset with only a single cross validation split.

Execute the commands below in an environment where you installed nnU-Net.

- Create a dataset.json file and place it in the dataset folder of the cross validation split you want to train.
An example dataset.json file can be found at [local_data/example_dataset.json](local_data/example_dataset.json),
Replace ``` numTraining ``` by the number of files in the ```imagesTr``` subdirectory and replace 
``` name ``` by the directory name of your nnU-Net dataset.

- nnU-Net internal preprocessing (this is different from the preprocessing step described above):
  ```bash
  nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
  ```
  With DATASET_ID the number of the nnU-Net dataset generated during preprocessing you want to use (check your nnUNet_raw
  folder). Use the 'trainval' dataset. This will create a new subdirectory in the ``` nnUNet_preprocessed ``` directory
  with the same DATASET_ID. 

- Next, copy the ``` splits_final.json ``` file generated during preprocessing to the subdirectory in 
``` nnUNet_preprocessed ``` corresponding to the DATASET_ID. By default, the ``` splits_final.json ``` 
file is generated at ``` nnunet/preprocessing_output/splits_final.json  ```.
This will tell nnU-Net which files to use for training and which files to use for validation. 
<br /> NOTE: if you do not do this, the nnU-Net will default to creating 10 random cross validation splits itself 
inside the dataset subdirectory, thus leading to unwanted behaviour.

- Train the model:

  ```bash
  CUDA_VISIBLE_DEVICES=x nnUNetv2_train DATASET_ID 2d 0 --npz
  ```
  With ``` x ``` the number of the GPU you want to use and ```DATASET_ID ``` the number of the nnU-Net dataset.
  This will create a new subdirectory in your ``` nnUNet_results ``` directory. You can track progress by looking at
  ``` nnUNetTrainer__nnUNetPlans__2d/fold_0/progress.png``` inside this subdirectory.
  The default will train for 1000 epochs, which is typically way too much. 
  However, for reproducibility, I recommend not changing this. Note that we explicitly train on fold 0, which is the
  only fold in the dataset because we created a separate dataset for each cross validation split.

## Run nnU-Net inference on test set
For a full explanation on how to use and configure nnU-Net, see
[https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md).

Execute the commands below in an environment where you installed nnU-Net. 

- Run nnU-Net '_find_best_configuration_'.
  ```bash
    nnUNetv2_find_best_configuration DATASET_ID -f 0
  ```
    where ``` DATASET_ID ``` is the number of the nnU-Net dataset you finished training for. This will create
    two files in the subdirectory of your dataset in your ``` nnUNet_results ``` directory:
    ```inference_instructions.txt ``` and ```inference_information.json ```. Normally, nnU-Net would check each 
    cross validation split to determine what ensemble to use based on validation performance. However, since we 
    explicitly created a separate dataset for each split, this will not happen. The second part of 
    '_find_best_configuration_' is to determine whether or not selecting the largest component of each segmentation
    improves performance, and if so, nnU-Net will also do this during inference.
- Follow the instructions in the ```inference_instructions.txt ``` file. Replace
  ``` INPUT_FOLDER ``` by the directory containing the test images generated during preprocessing,
  ``` OUTPUT_FOLDER ``` by the directory where the predictions should be saved (you can choose this yourself), and
  ``` OUTPUT_FOLDER_PP ``` by the directory where the postprocessed predictions should be stored (you can choose this
  yourself).
  


## Evaluate

TODO