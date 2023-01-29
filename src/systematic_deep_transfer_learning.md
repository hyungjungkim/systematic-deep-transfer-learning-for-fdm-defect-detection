# Systematic deep transfer learning
## Idea
This code contains a systematic deep transfer learning method based on a small dataset to detect defects of the spaghetti-shape error in the FDM process. First, images for training and validation are augmented effectively by considering a variation in part shape and visual monitoring conditions. Second, the feature extractors are divided into multiple training sections to evaluate various fine-tuning strategies systematically.

![](/src/systematic_deep_transfer_learning.jpg)

To get more detailed information, please refer to the following published paper. It's open-access.
- Kim, H., Lee, H., & Ahn, S. H. (2022). **Systematic deep transfer learning method based on a small image dataset for spaghetti-shape defect monitoring of fused deposition modeling**. Journal of Manufacturing Systems, 65, 439-451. [[Link](https://doi.org/10.1016/j.jmsy.2022.10.009)] 

## Dependencies
|Item|Requirement|
|---|---|
|Language|Python 3.7 or higher|
|Image augmentation|OpenCV 0.0.0 or higher|
|Deep learning - Modeling|Keras 0.0.0 or higher|
|Deep learning - Training|Tensorflow 2.3.0 or higher|
|Others|Scikit-learn|
