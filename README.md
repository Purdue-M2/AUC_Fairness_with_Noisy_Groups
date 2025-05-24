# Preserving AUC Fairness in Learning with Noisy Protected Groups
Mingyang Wu, Li Lin, Wenbin Zhang, Xin Wang, Zhenhuan Yang, Shu Hu
---

This repository is the official implementation of our paper **"Preserving AUC Fairness in Learning with Noisy Protected Groups"**, which has been accepted by ICML 2025.

## 1. Installation

You can set up the environment using the following commands:

```bash
cd AUC-Fairness-Noisy
conda create -n FairnessNoisy python=3.9.0
conda activate FairnessNoisy
pip install -r requirements.txt
```

## 2. Dataset Preparation

We share the tabular dataset Adult, Bank, and Default datasets with demographic annotations from [paper](https://arxiv.org/pdf/2208.10451), and image datasets FF++, Celeb-DF, DFD, and DFDC datasets with demographic annotations from [paper](https://arxiv.org/pdf/2208.05845.pdf), which can be downloaded through this [link](https://purdue0-my.sharepoint.com/:f:/g/personal/lin1785_purdue_edu/EtMK0nfxMldAikDxesIo6ckBVHMME1iIV1id_ZsbM9hsqg?e=WayYoy).

You can also access these **re-annotated** image datasets with prediction uncertainty scores via our **[AI-Face-FairnessBench](#).**

Alternatively, you can download these datasets from their official sources and process them by following these steps:

- **Download** [FF++](https://github.com/ondyari/FaceForensics), [Celeb-DF](https://github.com/yuezunli/celeb-deepfakeforensics), [DFD](https://ai.googleblog.com/2019/09/contributing-data-to-deepfake-detection.html), and [DFDC](https://ai.facebook.com/datasets/dfdc/) datasets.
- **Download annotations** for these datasets according to [this paper](https://arxiv.org/pdf/2208.05845.pdf) and their [code](https://github.com/pterhoer/DeepFakeAnnotations), then extract the demographic information from all images.
- **Extract, align, and crop faces** using [DLib](https://www.jmlr.org/papers/volume10/king09a/king09a.pdf), and save them to:
  ```bash
  /path/to/cropped_images/
  ```
- **Split cropped images into train/val/test with a ratio of 60%/20%/20%, ensuring no identity overlap.

## 3. Load Pretrained Weights

Before running the training code, make sure you load the **pre-trained weights**. We provide pre-trained weights under  
[pretrained](https://drive.google.com/drive/folders/1mqTiQ14pyXeLDX1EUibnqP2EWE-504oY?usp=sharing). You can also download the **Xception model** trained on ImageNet (through this [link](#)) or use your own pretrained **Xception**.

---

## 4. Train

To run the training code, first navigate to the `./training/` folder, then execute the following command to train our detector **with** loss flattening strategy:
For image datasets:

```bash
cd training
python train_images.py
--lr: learning rate, default is 0.0005.
--train_batchsize: batch size, default is 32.
--seed: random seeds
--datapath: /path/to/train.csv
--model: detector name: auc_effinet
--device: gpu ids for training.
--num_groups: default is 4
--method: ours is 'auc'
--backbone: default is 'efficientnetb4'
--pho: learning parameter for SAM, default is 0.5
--ratio: noise ratio, default is 0.02
```

For tabular datasets:

```bash
cd training
python train_tabular.py
--lr: learning rate, default is 0.01.
--train_batchsize: batch size, default is 10000.
--seed: random seeds
--device: gpu ids for training.
--pho: learning parameter for SAM, default is 0.0005
--method: ours is 'auc'
--dataset: training dataset name: default.
--ratio: noise ratio, default is 0.02
```


## 5. Test
For model testing, we provide a python file to test our model by running python test.py.

```bash
--checkpoints : /path/to/saved/model.pth
--model: 'auc_effinet'
--device: gpu ids for training.
--inter_attribute : intersectional group names divided by '-': male-female
--label_attribute : label attribute name divided by '-': pos-neg
--test_datapath : /path/to/test.csv
--savepath : /where/to/save/predictions.npy(labels.npy)/results/
--model_structure : backbone: auc_effinet.
--test_batch_size : testing batch size: default is 32.
```

## Citation
Please kindly consider citing our papers in your publications.


