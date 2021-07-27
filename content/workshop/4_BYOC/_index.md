+++
title = "Use a custom Docker image"
date = 2020-04-27T08:16:25-06:00
weight = 004
chapter = true
type = "mylayout"
+++

## Bring Your Own Containers

Amazon SageMaker's built-in algorithms and supported frameworks should cover most use cases, but there are times when you may need to use an algorithm from a package not included in any of the supported frameworks. You might also have a pre-trained model picked or persisted somewhere which you need to deploy. SageMaker uses Docker images to host the training and serving of all models, so you can supply your own custom Docker image if the package or software you need is not included in a supported framework. This may be your own Python package or an algorithm coded in a language like Stan or Julia. For these images you must also configure the training of the algorithm and serving of the model properly in your Dockerfile. This requires intermediate knowledge of Docker and is not recommended unless you are comfortable writing your own machine learning algorithm. Your Docker image must be uploaded to an online repository, such as the Amazon Elastic Container Registry (ECR) before you can train and serve your model properly.

For more information on custom Docker images in SageMaker, see [Using Docker containers with SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/docker-containers.html).

source: https://docs.aws.amazon.com/sagemaker/latest/dg/algorithms-choose.html#custom-image-use-case