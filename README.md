<!-- TODO

-->

# ProFD: Prompt-Guided Feature Disentangling for Occluded Person Re-Identification

**A strong baseline for body part-based person re-identification** 

[[Paper](https://openreview.net/pdf?id=o2axlPlXYY)] [[Video]] [[Poster]]

## News
- [2024.09.14] We release the first version of our codebase. Please update frequently as we will add more documentation during the next few weeks.

## Instructions
### Installation
Make sure [conda](https://www.anaconda.com/distribution/) is installed.

    # clone this repository
    git clone https://github.com/VlSomers/bpbreid

    # create conda environment
    cd bpbreid/ # enter project folder
    conda create --name bpbreid python=3.10
    conda activate profd
    
    # install dependencies
    # make sure `which python` and `which pip` point to the correct path
    pip install -r requirements.txt
    
    # install torch and torchvision (select the proper cuda version to suit your machine)
    conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
    
    # install torchreid (don't need to re-build it if you modify the source code)
    python setup.py develop
    

### Download human parsing labels
You can download the human parsing labels on [GDrive](https://drive.google.com/drive/folders/1IbCAbjj3XtV3_tFOsCuqBi79ZiDqNc1H?usp=sharing). 
These labels were generated using the [PifPaf](https://github.com/openpifpaf/openpifpaf) pose estimation model and then filtered using segmentation masks from [Mask-RCNN](https://github.com/facebookresearch/detectron2).
We provide the labels for five datasets: **Market-1501**, **DukeMTMC-reID**, **Occluded-Duke**, **Occluded-ReID** and **P-DukeMTMC**.
After downloading, unzip the file and put the `masks` folder under the corresponding dataset directory.
For instance, Market-1501 should look like this:

    Market-1501-v15.09.15
    ├── bounding_box_test
    ├── bounding_box_train
    ├── masks
    │   └── pifpaf_maskrcnn_filtering
    │       ├── bounding_box_test
    │       ├── bounding_box_train
    │       └── query
    └── query

Make also sure to set `data.root` config to your dataset root directory path, i.e., all your datasets folders (`Market-1501-v15.09.15`, `DukeMTMC-reID`, `Occluded_Duke`, `P-DukeMTMC-reID`, `Occluded_REID`) should be under this path.
We plan to add automatic download of these labels in the future.
We also plan to release the python script to generate these labels for any given dataset.


### Generate human parsing labels

You can create human parsing labels for your own dataset using the following command:

    conda activate profd
    python scripts/get_labels --source [Dataset Path] 

The labels will be saved under the source directory in the *masks* folder as per the code convention.


### Inference
You can test the above downloaded models using the following command:

    conda activate profd
    python scripts/main.py --config-file configs/clipreid/clipreid_<target_dataset>_test.yaml
    
For instance, for the Market-1501 dataset:

    conda activate profd
    python scripts/main.py --config-file configs/clipreid/clipreid_market1501_test.yaml
    
Configuration files for other datasets are available under `configs/clipreid/`.
Make sure the `model.load_weights` in these `yaml` config files points to the pre-trained weights you just downloaded. 

### Training
Training configs for five datasets (Market-1501, DukeMTMC-reID, Occluded-Duke, Occluded-ReID and P-DukeMTMC) are provided in the `configs/clipreid/` folder. 
A training procedure can be launched with:

    conda activate profd
    python ./scripts/main.py --config-file configs/clipreid/clipreid_<target_dataset>_train.yaml
    
For instance, for the Occluded-Duke dataset:

    conda activate profd
    python scripts/main.py --config-file configs/clipreid/clipreid_occ_duke_train.yaml

Make sure to download and install the human parsing labels for your training dataset before runing this command.


## Questions and suggestions
If you have any question/suggestion, or find any bug/issue with the code, please raise a GitHub issue in this repository, I'll be glab to help you as much as I can!
I'll try to update the documentation regularly based on your questions. 


## Citation
If you use this repository for your research or wish to refer to our method [ProFD](https://openreview.net/pdf?id=o2axlPlXYY), please use the following BibTeX entry:
```
@inproceedings{cui2024profd,
  title={ProFD: Prompt-Guided Feature Disentangling for Occluded Person Re-Identification},
  author={Cui, Can and Huang, Siteng and Song, Wenxuan and Ding, Pengxiang and Min, Zhang and Wang, Donglin},
  booktitle={ACM Multimedia 2024},
  year={2024}
}
```

## Acknowledgement
This codebase is a fork from [Torchreid](https://github.com/KaiyangZhou/deep-person-reid) and [BPBReID](https://github.com/VlSomers/bpbreid)


