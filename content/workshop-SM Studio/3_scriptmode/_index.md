+++
title = "SageMaker Script Mode"
date = 2020-04-27T08:16:25-06:00
weight = 003
chapter = true
type = "mylayout"
+++

## Bring your own model with Amazon SageMaker script mode

As the prevalence of machine learning (ML) and artificial intelligence (AI) grows, you need the best mechanisms to aid in the experimentation and development of your algorithms. You might begin with the several built-in algorithms in Amazon SageMaker that simply require you to point the algorithm at your data and start a SageMaker training job. At the other end of the spectrum, you might be quite specialized and have several highly customized algorithms and Docker containers to support those algorithms, and AWS has a workflow to create and support these bespoke components as well. However, it’s increasingly common to have invested time and energy into researching, testing, and building several custom ML algorithms, but use widely used frameworks such as scikit-learn. In this scenario, you don’t need or want to invest the time, money, and resources to create and support bespoke containers.

To address this, SageMaker offers a solution using script mode. Script mode enables you to write custom training and inference code while still utilizing common ML framework containers maintained by AWS. Script mode is easy to use and extremely flexible. In this post, we discuss three primary use cases for using script mode, and how script mode can accelerate your algorithm development and testing while simultaneously decreasing the amount of time, effort, and resources required to bring your custom algorithm to the cloud.


source: https://aws.amazon.com/blogs/machine-learning/bring-your-own-model-with-amazon-sagemaker-script-mode/