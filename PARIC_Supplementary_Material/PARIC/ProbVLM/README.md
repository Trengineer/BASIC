## ProbVLM
### Repository Structure
The folder structure is primarily based on the original ProbVLM repository. Below, we outline only the files and directories that have been modified to align with our framework.
```
ProbVLM/
├── ckpt/ (gets created after running the training scripts)
├── Datasets/(created as stated in the main README file of the repo)
├── src/
│   ├── ds/
│   │   ├── annotations/
│   │   │   ├── cub/
│   │   │   │   └── seen_test_images.txt  (removed overlapping instances between test and val)
│   │   │   └── coco/
│   │   │       ├── coco_train_ids_extra.npy  
│   │   │       └── coco_dev_ids_extra.npy
│   │   ├── __init__.py  
│   │   ├── _dataloader.py  
│   │   ├── _dataloader_extra.py  
│   │   ├── cub.py
|   |   ├── food.py
│   │   └── coco.py  
│   ├── Attention_comparison.ipynb  (extracts attention maps for comparison)
│   ├── COCO_extend_anotations.ipynb  (extends annotation files and ids for COCO to generate the npy files in annotations/coco)
│   ├── Confusion_matrices.ipynb  (evaluates results)
│   ├── CUB_metadata_fix.ipynb  (aligns dataset splits for Waterbirds between GALS and ProbVLM)
│   ├── train_ProbVLM_CLIP.ipynb  
│   ├── CUB_extend_annotations.ipynb  (extends the annotation files for Waterbirds)
│   ├── Attention_maps_COCO_Python.py  (extracts attention maps for COCO, checkpoints from training are needed)
│   ├── Attention_maps_CUB_Python.py  (extracts attention maps for Waterbirds, checkpoints from training are needed)
│   ├── Attention_maps_FOOD_MEAT_Python.py (extracts attention maps for meat, checkpoints from training are needed)
│   ├── Attention_maps_FOOD_RED_Python.py (extracts attention maps for red meat, checkpoints from training are needed)
│   ├── Attention_maps_WATERBIRDS_95_Python.py (extracts attention maps for Waterbirds 95, checkpoints from training are needed)
│   ├── attention_utils_p.py  
│   ├── networks.py  
│   ├── train_probVLM.py
│   ├── train_probVLM_FOOD.py 
│   ├── train_ProbVLM_CLIP-RN50-CUB.py  (training script for Waterbirds)
│   ├── train_ProbVLM_CLIP-RN50.py  (training script for COCO)
│   ├── train_ProbVLM_CLIP-RN50-FOOD.py (training script for MEAT and Red meat)
│   ├── train_ProbVLM_CLIP-RN50-WATERBIRDS_95.py (training script for Waterbirds 95)
│   └── utils.py

```
      
