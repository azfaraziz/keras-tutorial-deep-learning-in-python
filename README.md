# Keras Tutorial

Following tutorial from: https://elitedatascience.com/keras-tutorial-deep-learning-in-python

## Install

This is done through MacOS running High Sierra 10.13.4

Install Miniconda for package management
https://conda.io/miniconda.html

Ran into error with stdio.h missing, to resolve this, execute the following

``` bash
xcode-select --install
```

``` bash
conda install theano pygpu
conda install keras
conda install matplotlib
```

Change the backend to use theano

``` bash
cp ~/.keras/keras.json ~/.keras/keras.json.bk
vi ~/.keras/keras.json
```

- Change "backend" to "theano"

## Tutorial

Follow the jupyter notebook: ```keras_cnn_example.ipynb```
or the extract ```keras_cnn_example_nb.py```