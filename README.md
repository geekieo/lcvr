# Large-scale Content-Only Video Recommendation
## Requirment
tensorflow

## Overview of Files

### lcvr 
Training Part
#### Training
* `train.py`: The primary script for training models.
* `losses.py`: Contains definitions for loss functions.
* `models.py`: Contains the base class for defining a model, visual similar network model.
* `readers.py`: Contains definitions for the Video dataset and Frame dataset readers.

#### Misc
* `config.py`: Configration
* `env_utils.py`: Functions for identifying different environments.
* `utils.py`: Common functions for training.
* `logger.py`: logging

## References
1. LCVR paper: J. Lee; S. Abu-El-Haija. [Large-Scale Content-Only Video Recommendation](http://www.joonseok.net/papers/deeprecs.pdf). ICCV 2017.  
1. CDML for video understanding paper: J. Lee, S. Abu-El-Haija, B. Varadarajan, A. Natsev. [Collaborative Deep Metric Learning for Video Understanding](http://www.joonseok.net/papers/cdml.pdf). KDD 2018.  
1. Youtube-8m dataset homepage: [https://research.google.com/youtube8m/](https://research.google.com/youtube8m/)  
1. Youtube-8m paper: S. Abu-El-Haija, N. Kothari, J. Lee, P. Natsev. [Youtube-8m: A large-scale video classification benchmark](https://arxiv.org/abs/1609.08675). 27 Sep 2016.  
1. Youtube-8m starter code: [https://github.com/google/youtube-8m](https://github.com/google/youtube-8m)  
1. Triplet loss blog: [Triplet Loss and Online Triplet Mining in TensorFlow](https://omoindrot.github.io/triplet-loss)  
1. Triplet loss paper: A. Hermans, L. Beyer, B. Leibe. [In Defense of the Triplet Loss for Person Re-Identification](https://arxiv.org/abs/1703.07737). arXiv:1703.07737v4  
1. Triplet loss project in TensorFlow : [https://github.com/omoindrot/tensorflow-triplet-loss](https://github.com/omoindrot/tensorflow-triplet-loss)  
