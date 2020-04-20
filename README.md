# Top-Down Networks
Code for reproducing the experiments of the paper [Top-Down Networks: A coarse-to-fine reimagination of CNNs](https://arxiv.org/abs/2004.07629). In case of any bugs/improvements, please reach Ioannis Lelekas ([giannislelekas@gmail.com](mailto:giannislelekas@gmail.com)).

## Requirements and Dependencies:
We advise Anaconda for quick installation of all requirements and dependencies. Simply use the provided `requirements.yml` and `conda env create -f requirements.yml`.

## Organization:
This repository is organized as follows:
```
src/                #source code
  lib/              #helper functions
  models/           #network architectures
  scripts/          #scripts for running the experiments

output/             #output folder; generated after running the code
  adversarial/      #extracted adversarial attacks
  gradcam/          #gradcam heatmaps
  graphs/           #training curves
  history/          #training history (loss, acc, learning curve)
  models/           #model checkpoints
  output/           #generated output from training

notebooks/          #notebooks for visualizing results
```

## Usage:
All scripts for running the experiments are in `src/scripts/`. Command line inputs are given as comments within the scripts.

## Citation:
For citing Top-Down Networks, please use the following:
```
@inproceedings{lelekas2020,
  title={Top-Down Networks: A coarse-to-fine reimagination of CNNs},
  author={I Lelekas and N Tomen and SL Pintea and JC van Gemert},
  journal={CVPR 2020 Workshop on Deep Vision},
  year={2020}
}
```

## Acknowledgements:
- https://github.com/BIGBALLON/cifar-10-cnn
- https://github.com/bethgelab/foolbox
- https://github.com/eclique/keras-gradcam
- https://github.com/koshian2/PCAColorAugmentation
