---
title: "Setup"
chapter: false
weight: 20
---


1. Login into [Amazon SageMaker](http://aws.amazon.com/sagemaker)
2. Select a Region.  
    * US East (N. Virginia), us-east-1

    To change the Region in the Amazon SageMaker console, use the Region selector at the upper-right corner of the console.

3. Choose `Notebook instances`, then choose `Create notebook instance`.

4. On the `Create notebook instance` page, provide the following information:

    * For `Notebook instance name`, type a name for your notebook instance.

    For `Notebook instance type`, choose `ml.m4.4xlarge`.

    * For IAM role, choose Create a new role. Amazon SageMaker creates an IAM role named `AmazonSageMaker-ExecutionRole-YYYYMMDDTHHmmSS`. The AWS managed policy `AmazonSageMakerFullAccess` is attached to the role. The role provides permissions that allow the notebook instance to call Amazon SageMaker and Amazon S3. Select `Any S3 bucket` for this workshop, and then `Create role`.

    * Choose Create notebook instance.

5. When the status of the notebook instance is `InService`, in the console, the notebook instance is ready to use. Choose `Open JupyterLab` to open the JupyterLab dashboard. The dashboard provides access to your notebook instance. 

In the next step, we will load the notebooks into Amazon SageMaker.
