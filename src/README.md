# Coding task

In this coding task, you'll have to build the inference pipeline for an image classification task on a well-known dataset. The dataset images are provided within this bundle, in the `/dataset` directory. The corresponding labels are serialized in `labels.json`.

### EfficientNet

The model whose performances we want to assess is EfficientNet (https://ai.googleblog.com/2019/05/efficientnet-improving-accuracy-and.html). We recommend to load the trained model using this package: https://pypi.org/project/efficientnet-pytorch/ and to use the `b0` version of the model. This specific model is already trained on the `ImageNet` dataset, no model training is involved in this exercise.

### Inference, metrics and analyzis

Once the inference pipeline is working, we ask you to provide a critical and creative analysis of the inference results obtained (e.g. metrics, imbalance, ...).
