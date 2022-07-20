# The official implementation of "Hierarchical Latent Structure for Multi-Modal Vehicle Trajectory Forecasting" accepted to ECCV22.
![](GIF/HLS_new.gif)

## Setup
 
+ **Dataset** : Create folders 'preprocessed_dataset/2sec_6sec' and 'preprocessed_dataset/2sec_3sec' in 
'NuscenesDataset' and 'ArgoverseDataset' folders, respectively. Next, download the preprocessed dataset files from https://drive.google.com/file/d/1iMfv3NdfUaqYvwX1pgFDWUq6UDdMBAAM/view?usp=sharing and copy them into the created folders, respectively. If you want to build new preprocessed dataset files from nuScenes and Argoverse Forecast datasets, download raw data from the corresponding websites and edit the parameter 'dataset_path' in 'config/config.json' file. Here, 'dataset_path' must point to the directory where the downloaded raw data exists. 


+ **Implemenation Environment** : The model is implemented by using Pytorch. We share our anaconda environment in the folder 'anaconda_env'.


## Train and Test New Models
To train the model from scratch, run the followings. The network parameters of the trained models will be stored in the folder ***saved_models***.
```sh
$ sh nuscenes_train.sh
$ sh argoverse_train.sh
```

**argumentparser.py** have a number of command-line flags that you can use to configure the model architecture, hyperparameters, and input / output settings. You can find the descriptions in the file.


To test the trained model, first edit the parameter 'exp_id' in 'nuscenes_test.sh' and 'argoverse_test.sh' files to match your experiment id and run the followings.
```sh
$ sh nuscenes_test.sh
$ sh argoverse_test.sh
```

## Test Pre-trained Models
To test the pre-trained models, first download the pre-trained model parameters from https://drive.google.com/file/d/1kEI3jLueqVejvim_Moh4909yBFQG4jaF/view?usp=sharing. Next, copy them into 'saved_models' folder. Finally, edit the parameter 'exp_id' in 'nuscenes_test.sh' and 'argoverse_test.sh' files to match the downloaded experiment id and run the followings.
```sh
$ python nuscenes_test.sh
$ python argoverse_test.sh
```

## Paper Download
Arxiv : https://arxiv.org/abs/2207.04624

## Citation
```
@InProceedings{Choi,
 author = {D. Choi and K. Min},
 title = {Hierarchical Latent Structure for Multi-Modal Vehicle Trajectory Forecasting},
 booktitle = {Eur. Conf. Comput. Vis.},
 year = {2022}
}
```
