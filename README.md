# ImageAEOT

This repository contains data and code for the paper, "Predicting Cell Lineages using Autoencoders and Optimal
Transport." ([link]())

## Data
The image and auxiliary files can be downloaded from Google Drive. ([link](https://drive.google.com/drive/folders/1lcYJF1iW3XJ9mAkby7GsONL1XxKGIVsA?usp=sharing))

## Setup and requirements
Dependencies are listed in environment.yml file and can be installed using Anaconda/Miniconda:
```
conda env create -f environment.yml
```
Autoencoder models were trained on an NVIDIA GTX 1080TI GPU. 

## Usage

To train the autoencoder on the coculture image files:
```
python run_train.py --datadir <path/to/image/directory> --save-dir <path/to/save/directory>
```

To train the autoencoder with latent space classifier on the coculture image files:
```
python run_train.py --datadir <path/to/image/directory> --save-dir <path/to/save/directory> --train-metafile splits/train_total_labeled.csv --val-metafile splits/val_total_labeled.csv --model-type AugmentedAE --dataset-type labeled
```

To extract AE features:
```
python get_features.py --datadir <path/to/image/directory> --save-dir <path/to/save/directory> --pretrained-file <path/to/checkpoint.pth> --ae-features
```

To extract AE features trained with latent space classifier:
```
python get_features.py --datadir <path/to/image/directory> --save-dir <path/to/save/directory> --pretrained-file <path/to/checkpoint.pth> --ae-features --model-type AugmentedAE
```

To run and evaluate features on the benchmark task:
```
python run_ot.py --featfile <path/to/features/file.txt> --evalfeatfile <path/to/label/file.txt> --save-dir <path/to/save/directory> --label1 <0/1/2> --label2 3 --reg .05
```
where the label files are provided found in the ```labels``` directory. If evaluating eccentricity or roundness, make sure to include the ```--split-features``` tag.
