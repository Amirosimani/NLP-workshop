---
title: "Lab: Amazon Sagemaker Model Monitor"
chapter: false
weight: 515
---

### Model Monitoring lab instructions
 
### PART 1 : Capturing real-time inference data from Amazon SageMaker endpoints

1. First, in the left side File Browser go to MLAI-> model-monitor ->  -> SageMaker-ModelMonitoring.ipynb. Choose the Python 3(Data Science) kernel.

![Model monitor start notebook](/images/model-monitor-start-notebook.png)

2. We use a pre-trained XGBoost Churn Prediction Model and Upload the pre-trained model to Amazon S3.


![Model monitor upload](/images/model-monitor-upload-model.png)

3. Deploy this model to Amazon SageMaker.

![Model monitor deploy](/images/model-monitor-deploy.png)

4. ‘DataCaptureConfig’ is required to capture model quality metrics.

![Model monitor DataCaptureConfig](/images/model-monitor-datacapture.png)

5. Lets now invoke the endpoint and start capturing the model quality metrics.

### PART 2: Model Monitor - baselining and continuous monitoring

In addition to collecting the data, Amazon SageMaker provides the capability for you to monitor and evaluate the data observed by the endpoints. For this:

1. Create a baseline with which you compare the real-time traffic.
2. Once a baseline is ready, setup a schedule to continuously evaluate and compare against the baseline.

#### 1. Constraint suggestion with baseline/training dataset: 
The training dataset with which you trained the model is usually a good baseline dataset. From this training dataset you can ask Amazon SageMaker to suggest a set of baseline `constraints` and generate descriptive `statistics` to explore the data.

1. Upload data 

![Model monitor upload](/images/model-monitor-upload-data.png)

2.  Suggest_baseline() function generates the baseline constraints.

![Model monitor baseline](/images/model-monitor-baseline.png)

#### 2. Analyzing collected data for data quality issues:
When you have collected the data above, analyze and monitor the data with hourly Monitoring Schedules

1. CronExpressionGenerator class will create an hourly schedule

![Model monitor cron schedule](/images/model-monitor-cron-schedule.png)

2. See violations compred to baseline

![Model monitor cron schedule](/images/model-monitor-violations.png)