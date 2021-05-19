---
title: "Lab: TF Inference (Optional)"
chapter: false
weight: 315 
---

### SageMaker Inference

This notebook shows how you can:

1. We deploy a TensorFlow Serving model to a hosted Endpoint for real-time inference. Additionally, we will discuss why and how to use Amazon Elastic Inference with the hosted endpoint..
2. We'll run a Batch Transform job using our data processing script and GPU-based Amazon SageMaker Model. More specifically, we'll perform inference on a cluster of two instances. The objects in the S3 path will be distributed between the two instances

Let's open our notebook and begin the lab:

1. Access the notebook instance you created earlier.
2. On the left sidebar, click the Folder icon and navigate into the `MLAI-workshop-resources/inference/tensorflow_cifar-10_with_inference_script/` folder. Double-click on `tensorflow-serving-cifar10.ipynb` to open it.
3. You are now ready to begin the notebook. Follow through the provided steps to build your custom container, and then leverage it for training in SageMaker.
