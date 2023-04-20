# Supervised learning in RKHM
Code for "[Learning in RKHM: a C*-Algebraic Twist for Kernel Machine](https://proceedings.mlr.press/v206/hashimoto23a.html)" by Yuka Hashimoto, Masahiro Ikeda, and Hachem Kadri

## Setup

To run the code, please install the following packages with Python 3.9:
- numpy
- tensorflow 2.6
- idx2numpy


## Data

For the experiment with MNIST, we need the dataset "train-images.idx3-ubyte" and its label "train-labels.idx1-ubyte", which can be downloaded from "http://yann.lecun.com/exdb/mnist/". Download these files to the same directry as that containing codes.


## Running the code

- For regression problem of synthetic data, run "python rkhm_syn.py".
- For noise reduction of MNIST, run "python rkhm_cnn_mnist.py".
