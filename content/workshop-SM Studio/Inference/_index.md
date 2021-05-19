+++
title = "Inference on SageMaker"
date = 2020-06-27T08:16:25-06:00
weight = 400
type = "mylayout"
chapter = true
+++

## Machine Learning Inference

To get predictions, we deploy your model. The method you use depends on how you want to generate inferences:

1. To get one inference at a time in real time, set up a persistent endpoint using Amazon SageMaker hosting services.

2. To get inferences for an entire dataset, use Amazon SageMaker batch transform.

#### Persistent endpoint
Amazon SageMaker lets you deploy a model to a hosted Endpoint for real-time inference.

#### Batch transform
To get inferences for an entire dataset, use batch transform. With batch transform, you create a batch transform job using a trained model and the dataset, which must be stored in Amazon S3. Amazon SageMaker saves the inferences in an S3 bucket that you specify when you create the batch transform job. Batch transform manages all of the compute resources required to get inferences.

Use batch transform when you:

* Want to get inferences for an entire dataset and index them to serve inferences in real time

* Don't need a persistent endpoint that applications (for example, web or mobile apps) can call to get inferences

* Don't need the subsecond latency that Amazon SageMaker hosted endpoints provide