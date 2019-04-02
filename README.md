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
```

### Dataset.
1. Generate the dataset for STVR.
    ```
    python gen_subsets.py
    ```
