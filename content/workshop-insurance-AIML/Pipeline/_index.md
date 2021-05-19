+++
title = "MLOps"
date = 2020-06-27T08:16:25-06:00
weight = 5
type = "mylayout"
chapter = true
+++

## Amazon Sagemaker MLops

Data Scientists and ML developers need more than a Jupyter notebook to create a ML model, to test it, to put it into production and to integrate it with a portal and/or a basic web/mobile application, in a reliable and flexible way.

MLOps is a concept that can help you:

1. To quickly deploy a change that you make in your codes for example
2. To build a repeatable and reliable ML workflow
3. To track all the steps you have taken and keep the lineage of ML workflow


There are different tools and workflow managers to build a CI/CD and Orchestration pipeline:

1. You can create/operate an automated ML pipeline using a traditional CI/CD tool, called [CodePipeline](https://aws.amazon.com/codepipeline/), to orchestrate the ML workflow.

2. Another AWS service that can used for this purpose is [Step Functions](
https://aws-step-functions-data-science-sdk.readthedocs.io/en/latest/readmelink.html#getting-started-with-sample-jupyter-notebooks). In this link, you'll also find the documentation of the python library that can be executed directly from your Jupyter notebook.

3. [Apache AirFlow](https://airflow.apache.org/) is a poweful Open Source tool that can also be integrated with SageMaker. Curious? Just take a look on the [SageMaker Operators for AirFlow](https://sagemaker.readthedocs.io/en/stable/using_workflow.html).

4. If you have a Kubernetes cluster and want to integrate SageMaker to that and manage the ML Pipeline from the cluster. No problem, take a look on the [SageMaker Operators for Kubernetes](https://aws.amazon.com/blogs/machine-learning/introducing-amazon-sagemaker-operators-for-kubernetes/).

