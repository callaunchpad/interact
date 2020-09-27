# Interact: A Human Object Interaction Project

## Project Overview
Human-object interaction aims to understand the relationship between humans and objects in images, such as “riding a bike” or “cutting a birthday cake”. While it involves traditional computer vision tasks such as detecting and identifying objects in images, it also involves reasoning beyond perception, and the ability to integrate complex information about different humans, objects, and settings. This semester, we will be investigating different models that accomplish this task, from using traditional CNN/RNN approaches to Graph Neural Network models, and seeing what we can learn from their performance.

## Setting up this project

### Prerequisites
- Python 3.6
- Pytorch 1.1.0
- DGL 0.3

### Installation
1. Create a conda environment for this project.

    ```
    conda create -n Interact python=3.6
    ```

2. Clone this repository.   

    ```
    git clone https://github.com/BIGJUN777/VS-GATs.git
    ```
  
3. Install Python dependencies:   

    ```
    pip install -r requirements.txt
    ```

### Prepare Data
1. Download the original [HICO-DET](http://www-personal.umich.edu/~ywchao/hico/) dataset and put it into `datasets/hico`. That folder should now contain:
- `images/`
- `tools/`
- `anno.mat`
- `anno_bbox.mat`
2. Download the word2vec model on GoogleNews [here](https://github.com/tmikolov/word2vec) and run `make` to compile the word2vec tool. Note that `make` here uses `gcc` and will most likely only work on Ubuntu/Mac machines, so if you're using a Windows machine, you should run this step under WSL.
3. Move the entire word2vec repo into `datasets/word2vec`.
4. Download the processed data from [HICO-DET](https://pan.baidu.com/s/1uodk72pc-lJEvAJqGn0X0A) (password: *3rax*), extract them, and copy it all into `datasets/processed`. That folder should now contain a `hico/` subfolder which contains a series of json and hdf5 files.

### Training Convolutional Neural Networks
Note: To be implemented.
Choose script `cnn_train.py` or `cnn_trainval.py` to train the model on either the training set or the training + validation set. There are several options we can set when training the model. See `cnn_train.py` for more argument details. 

```
    python cnn_train.py/cnn_trainval.py
```

Note that this training should work on both GPU and CPU.

You can visualized the training process through tensorboard: `tensorboard --logdir='log/'`.

Checkpoints will be saved in `checkpoints/` folder.

### Testing Convolutional Neural Networks
Note: To be implemented.

Results will be saved in `result/` folder.

### Training Graph Neural Networks
Note: Currently only trains VS-GATS
Choose script `hico_train.py` or `hico_trainval.py` to train the model on either the training set or the training + validation set. There are several options we can set when training the model. See `hico_train.py` for more argument details. 

```
    python hico_train.py/hico_trainval.py --e_v='vs_gats_train' --t_m='epoch' --b_s=32 --f_t='fc7' --layers=1 --lr=0.00001 --drop_prob=0.3 --bias='true' --optim='adam' --bn=False --m_a='false' --d_a='false' --diff_edge='false' 
```

Note that this training should work on both GPU and CPU.

You can visualized the training process through tensorboard: `tensorboard --logdir='log/'`.

Checkpoints will be saved in `checkpoints/` folder.

### Testing Graph Neural Networks
Run the following script: option 'final_ver' means the name of which experiment and 'path_to_the_checkpoint_file' means where the checkpoint file is. 

```
bash hico_eval.sh 'final_ver' 'path_to_the_checkpoint_file'
```

Results will be saved in `result/` folder.

### Acknowledgement
Much of the data processing and GNN training/eval scripts are built upon [Visual-Semantic Graph Attention Network for Human-Object Interaction Detecion](https://github.com/birlrobotics/vs-gats). 
