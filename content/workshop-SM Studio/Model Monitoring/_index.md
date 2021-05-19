+++
title = "Model Monitoring"
date = 2020-06-27T08:16:25-06:00
weight = 500
type = "mylayout"
chapter = true
+++

## Model Monitoring 

Amazon SageMaker enables you to capture the input, output and metadata for invocations of the models that you deploy. It also enables you to analyze the data and monitor its quality. 

![Model monitor flow image](/images/model-monitor-workflow.png)

To enable model monitoring, you take the following steps, which follow the path of the data through the various data collection, monitoring, and analysis processes:

* Enable the endpoint to capture data from incoming requests to a trained ML model and the resulting model predictions.

* Create a baseline from the dataset that was used to train the model. The baseline computes metrics and suggests constraints for the metrics. Real-time predictions from your model are compared to the constraints, and are reported as violations if they are outside the constrained values.

* Create a monitoring schedule specifying what data to collect, how often to collect it, how to analyze it, and which reports to produce.

* Inspect the reports, which compare the latest data with the baseline, and watch for any violations reported and for metrics and notifications from Amazon CloudWatch.