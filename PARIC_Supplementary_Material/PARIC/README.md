# EGALS

NOTE: The method originally named EGALS has been updated to PARIC. However, due to time constraints, the code implementation still retains the original name EGALS and has not been updated to reflect the new name. In the final version, the code will be updated to replace all instances of EGALS with PARIC to ensure consistency. Wherever EGALS is currently used, it should be interpreted as PARIC.

### Setting up GALS

#### Environment
To build the environment needed to run GALS image classifier [ref], use the GALS/env.yaml and GALS/requirements.txt
#### Datasets
You will need to download the following datasets and organize them into the ./data folder.

- Waterbirds 100%: You'll first need to download the segmentations for original CUB_200_2011 from here: [segmentations](https://www.kaggle.com/datasets/wenewone/cub2002011?select=segmentations#:~:text=Animals-,segmentations,-(200%20directories)) and put it inside a folder called CUB_200_2011. Then, we use the dataset provided in the GALS Github: [Waterbirds-100%](https://github.com/spetryk/GALS?tab=readme-ov-file#:~:text=the%20dataset%20here%3A-,Waterbirds%2D100%25.,-Food101%3A%20Original).
- MSCOCO: [Original dataset page](https://cocodataset.org/#download). Please use the original dataset page to download the COCO 2014 train & validation images and annotations. We use MSCOCO-ApparentGender provided by GALS. Please download the files about the splits here: [MSCOCO-ApparentGender](https://github.com/spetryk/GALS?tab=readme-ov-file#:~:text=the%20splits%20here%3A-,MSCOCO%2DApparentGender,-.).
- Food-101: For the meat and red meat experiments, we use subsets of the [Food-101 dataset](https://www.kaggle.com/datasets/dansbecker/food-101). For the meat experiment, you can download the labels from here: [MEAT](https://drive.google.com/file/d/14GyaK0ybz1zH2rq5Oi40GYr5Z-sJsueK/view?usp=sharing), and [RED MEAT](https://drive.google.com/file/d/1buMB_vNNsNeEBnOqUzFQTO5q9_Q8hEh2/view?usp=sharing).
  
In the end the data folder should follow this structure:
```
./data
├── CUB_200_2011/
│   └── segmentations/  (segmentation folder for CUB from Kaggle)
├── waterbird_1.0_forest2water2/  (Waterbirds-100%)
├── COCO/
│   ├── annotations/  (COCO annotations from original dataset page)
│   ├── train2014/  (COCO images from original dataset page)
│   ├── val2014/  (COCO images from original dataset page)
│   └── COCO_gender/  (ApparentGender files)
├── food-101/
│   ├── images/ (Food 101 images)
│   ├── meta/  (Food 101 metadata + extra files created by substracting the classes for meat and red meat from the original train and test)
│   |   ├── labels-meat.txt
│   |   ├── labels-redmeat.txt
│   |   ├── test-meat.txt
│   |   ├── test-redmeat.txt
│   |   ├── train-meat.txt
│   |   └── train-redmeat.txt

  
```

### Setting up ProbVLM
#### Evironment
To build the environment needed to run ProbLVM probabilitic layers [ref], use the ProbVLM/env2.yaml and ProbVLM/requirements.txt. Additionally, outside of ProbVLM folder, in the main working directory, install [LAVIS](https://github.com/salesforce/LAVIS?tab=readme-ov-file#lavis---a-library-for-language-vision-intelligence:~:text=%5BCOMING%20SOON%5D-,Installation,-(Optional)%20Creating%20conda).

#### Datasets
You will need to download the following datasets and organize them into the ./Datasets folder.

- Waterbirds 100%: You'll first need to download the segmentations and annotations for original CUB_200_2011 from here: [CUB](https://www.kaggle.com/datasets/wenewone/cub2002011?select=segmentations#:~:text=200%2D2011%20Dataset-,CUB%2D200%2D2011,-arrow_drop_up), and put it inside the Datasets folder as shown below. Then, we use the dataset provided in the GALS Github: [Waterbirds-100%](https://github.com/spetryk/GALS?tab=readme-ov-file#:~:text=the%20dataset%20here%3A-,Waterbirds%2D100%25.,-Food101%3A%20Original).
- MSCOCO: [Original dataset page](https://cocodataset.org/#download). Please use the original dataset page to download the COCO 2014 train & validation images and captions.
  
In the end the Datasets folder should follow this structure:
```
./Datasets
├── CUB_200_2011/
│   └── segmentations/  (segmentation folder for CUB from Kaggle)
├── waterbird_1.0_forest2water2/  (Waterbirds-100%)
├── text_c10/
├── COCO/
│   ├── images/
│   │   └── total/  (COCO train and val2014 images combined in the same folder from original dataset page)
│   ├── captions_train2014.json
│   └── captions_val2014.json

```
For the meat and red meat experiments, all the code is build up refering to the data downloaded in GALS/data directory, so no need for extra folder.

### Training and Evaluation

#### Training

Training process has 3 stages:

1. Training probabilistic encoders: Run the scripts called train_ProbVLM_CLIP-RN50.py for COCO, train_ProbVLM_CLIP-RN50-CUB.py for Waterbirds 100, train_ProbVLM_CLIP-RN50-WATERBIRDS_95.py for Waterbirds 95, train_ProbVLM_CLIP-RN50-FOOD.py for meat and red meat.
2. Extract attention maps: Run the scripts called Attention_maps_COCO_Python.py for COCO, Attention_maps_CUB_Python.py for Waterbirds 100, Attention_maps_WATERBIRDS_95_Python.py for Waterbirds 95, Attention_maps_FOOD_MEAT_Python.py for meat and Attention_maps_FOOD_RED_Python.py for red meat.
3. Train guided image classifier: Run the following command changing the config files accordingly. The config files need to be modified depending on what approach is followed. For more information refer to the readme file inside GALS folder.
   ```
   CUDA_VISIBLE_DEVICES=0,1,2 python main.py --name name --config configs/approach
   ```
   The --name flag is used for Weights & Biases logging. You can add --dryrun to the command to run locally.
   

#### Evaluation

To evaluate run the following command changing the config files accordingly to the dataset, and provide a path to their trained checkpoint.
```
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/dataset_val.yaml --test_checkpoint checkpoint.ckpt
```
### Acknowledgements

We are extremly grateful to the following people, from which have inspried our work and from whom we have used code throughout this repository that is taken or based off of their work:

- S. Petryk, L. Dunlap, K. Nasseri, J. Gonzalez, T. Darrell, and A. Rohrbach: https://github.com/spetryk/GALS
- U. Upadhyay, S. Karthik, M. Mancini, and Z. Akata: https://github.com/ExplainableML/ProbVLM

