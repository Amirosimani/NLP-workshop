+++
title = "Hugging Face with SageMaker"
date = 2020-04-27T08:16:25-06:00
weight = 005
chapter = true
type = "mylayout"
+++

## Bring Your Own Containers

Amazon SageMaker enables customers to train, fine-tune, and run inference using Hugging Face models for Natural Language Processing (NLP) on SageMaker. You can use Hugging Face for both training and inference. This functionality is available through the development of Hugging Face AWS Deep Learning Containers. These containers include Hugging Face Transformers, Tokenizers and the Datasets library, which allows you to use these resources for your training and inference jobs. For a list of the available Deep Learning Containers images, see [Available Deep Learning Containers Images](https://github.com/aws/deep-learning-containers/blob/master/available_images.md). These Deep Learning Containers images are maintained and regularly updated with security patches.

To use the Hugging Face Deep Learning Containers with the SageMaker Python SDK for training, see the [Hugging Face SageMaker Estimator](https://sagemaker.readthedocs.io/en/stable/frameworks/huggingface/index.html). With the Hugging Face Estimator, you can use the Hugging Face models as you would any other SageMaker Estimator. However, using the SageMaker Python SDK is optional. You can also orchestrate your use of the Hugging Face Deep Learning Containers with the AWS CLI and AWS SDK for Python (Boto3).

For more information on Hugging Face and the models available in it, see the [Hugging Face documentation](https://huggingface.co/).