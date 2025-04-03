# dual-channel
 The lack of distortion information in images with unstable exposure areas or cluttered areas has been a
 major bottleneck in the development of NR-IQA methods. Traditional methods have not been able to recognize and
 extract the features in the image well, and the weight assignment of convolution kernel is also confusing, to solve
 this problems, we proposes a NR-IQA method based on mining hard samples (data with prediction error exceeding
 a specific threshold in the network pruning channel) and adaptive deformable convolution. The deviation learning
 property of deformable convolution is utilized to enable the model to adapt to different sizes of images and different
 shapes of predicted objects, after which the images are fed into the regular channel and network pruning channel
 respectively, the hard samples are filtered by the quality prediction difference, and the samples for the hard samples
 are re-allocated until the model training reaches the standard. Considering that the subjective quality score of an
 image is affected by both the content and structure of the image, a stepped feature fusion structure is added in the
 feature extraction stage of this model. In addition to this, in order to better utilize the performance of the hard sample
 mining module, as well as to learn more hard samples, we use a mixed dataset training strategy to expand the dataset
 size. Experimental results among several real distorted datasets and synthetic distorted datasets show that our model
 outperforms most of the current NR-IQA methods in terms of performance, and also achieves a large advantage in
 terms of generalization performance and adaptability to unfamiliar images.
 ![f0ccd1cadd65103ce74a7ccb23d6f495](https://github.com/user-attachments/assets/c48e7340-ff03-4f7e-a295-3c217299e8e5)
