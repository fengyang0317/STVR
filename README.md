# Spatio-temporal Video Re-localization by Warp LSTM
by [Yang Feng](http://cs.rochester.edu/u/yfeng23/), Lin Ma, Wei Liu, and
[Jiebo Luo](http://cs.rochester.edu/u/jluo)

### Introduction
We formulate a new task named spatio-temporal video re-localization. Given a
query video and a reference video, spatio-temporal video re-localization aims to
localize tubelets in the reference video such that the tubelets semantically
correspond to the query. For more details, please refer to our
[paper](https://arxiv.org/abs/1905.03922).

![alt text](http://cs.rochester.edu/u/yfeng23/cvpr19_ava/framework.png 
"Framework")

### Citation

    @InProceedings{feng2019spatio,
      author = {Feng, Yang and Ma, Lin and Liu, Wei and Luo, Jiebo},
      title = {Spatio-temporal Video Re-localization by Warp LSTM},
      booktitle = {CVPR},
      year = {2019}
    }

### Requirements
```
pip install tensorflow-gpu
sudo apt install python-opencv
```

In case you are only interested in the proposed [Warp LSTM][2], please find the
implementation in the link above.

### Dataset.
1. Generate the dataset for STVR.
   ```
   python gen_subsets.py
   ```

### Install Tensorflow Object Detection API.
2. 
    ```
    mkdir ~/workspace
    cd ~/workspace
    git clone https://github.com/tensorflow/models.git
    cd models
    git remote add yang https://github.com/fengyang0317/tf_models.git
    git fetch yang
    git checkout warp_c
    export PYTHONPATH=${HOME}/workspace/models/research/object_detection:\
    ${HOME}/workspace/models/research:\
    ${HOME}/workspace/models/research/slim
    ```
Then follow the instructions in [Installation][1]


### Extract video clips.
3. We cut the videos to one-second clip for loading into Tensorflow.
    ```
    cd ~/workspace
    git clone https://github.com/fengyang0317/STVR.git
    cd STVR
    python split_videos.py --data_dir PATH_TO_VIDEOS --subset train
    python split_videos.py --data_dir PATH_TO_VIDEOS --subset val
    ```

### Training.
4. 
    ```
    python main.py --data_dir PATH_TO_VIDEOS --batch_size 8 --i3d_ckpt CKPT_PATH
    ```

### Evaluation.
5.
    ```
    python eval.py --data_dir PATH_TO_VIDEOS --i3d_ckpt CKPT_PATH
    python compute_ap.py
    ```

### Credits
Part of the code is from 
[kinetics-i3d](https://github.com/deepmind/kinetics-i3d),
[ActivityNet](https://github.com/activitynet/ActivityNet/blob/master/Evaluation/get_ava_performance.py), and
[Tensorflow Object Detection API](
https://github.com/tensorflow/models/tree/master/research/object_detection).

[1]: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md
[2]: https://github.com/fengyang0317/tf_models/blob/warp_c/research/object_detection/meta_architectures/conv_lstm_cell.py
