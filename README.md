![sagnet](figs/sagnet.png)

# Fork Changes
### Summary
- Add argument `--reps` to rerun a the chosen experiment multiple times (default=5). 
- Training performance history are saved to .csv for each rep
- After training and test the models, the average test accuracy for 3 Model selections is reports:
    - Final model test accuracy
    - Best test accuracy (the highest test accuracy recorded in each training rep)
    - Best crossval test accuracy (the best model selected is that with the highest cross-val accuracy during each training rep)
- I will upload the training logs for each PACS split soon. 

### Requirements

- `python=3.8.10`
- `requirements.txt`
- Note: This uses PyTorch 1.4 -- PyTorch 1.7 or newer does not seem to work due to some change with the backward pass and retain_graph


### Updated Usage
e.g. the train command is:
```
python train.py --sources Rest --targets [domain] --method sagnet --sagnet --batch-size 32 -g [gpus] --reps 5
```

Example performance outputs for `--target art_painting`:
```
--- first input args are printed at start ---
Save directory: checkpoint/pacs/sagnet/cartoon-sketch-photo-0
Save directory: checkpoint/pacs/sagnet/cartoon-sketch-photo-1
Save directory: checkpoint/pacs/sagnet/cartoon-sketch-photo-2
Save directory: checkpoint/pacs/sagnet/cartoon-sketch-photo-3
Save directory: checkpoint/pacs/sagnet/cartoon-sketch-photo-4


Targets=['art_painting'] (5 Repeats):

Final Test Accuracy = 0.810

Best Test Accuracy each rep:
	 [0.82275390625, 0.830078125, 0.8310546875, 0.8427734375, 0.8359375]
Average = 0.833

Accuracy of best cross-val model each rep (usual PACS model selection):
	 [0.81298828125, 0.8076171875, 0.8212890625, 0.826171875, 0.8212890625]
Average = 0.818
```


# Style-Agnostic Networks (SagNets)
By Hyeonseob Nam, HyunJae Lee, Jongchan Park, Wonjun Yoon, and Donggeun Yoo.

Lunit, Inc.

### Introduction
This repository contains a pytorch implementation of Style-Agnostic Networks (SagNets) for Domain Generalization.
It is also an extension of our method which won the first place in Semi-Supervised Domain Adaptation of [Visual Domain Adaptation (VisDA)-2019 Challenge](https://ai.bu.edu/visda-2019/).
Details are described in [Reducing Domain Gap by Reducing Style Bias](https://openaccess.thecvf.com/content/CVPR2021/papers/Nam_Reducing_Domain_Gap_by_Reducing_Style_Bias_CVPR_2021_paper.pdf), **CVPR 2021 (Oral)**.

### Citation
If you use this code in your research, please cite:

```
@inproceedings{nam2021reducing,
  title={Reducing Domain Gap by Reducing Style Bias},
  author={Nam, Hyeonseob and Lee, HyunJae and Park, Jongchan and Yoon, Wonjun and Yoo, Donggeun},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2021}
}
```

### Prerequisites
- [PyTorch 1.0.0+](https://pytorch.org/)
- Python 3.6+
- Cuda 8.0+

### Setup
Download [PACS](http://www.eecs.qmul.ac.uk/~dl307/project_iccv2017) dataset into ```./dataset/pacs```
```
images ->  ./dataset/pacs/images/kfold/art_painting/dog/pic_001.jpg, ...
splits ->  ./dataset/pacs/splits/art_painting_train_kfold.txt, ...
```

### Usage
#### Multi-Source Domain Generalization
```
python train.py --sources Rest --targets [domain] --method sagnet --sagnet --batch-size 32 -g [gpus]
```
#### Single-Source Domain Generalization
```
python train.py --sources [domain] --targets Rest --method sagnet --sagnet --batch-size 96 -g [gpus]
```
Results are saved into ```./checkpoint```
