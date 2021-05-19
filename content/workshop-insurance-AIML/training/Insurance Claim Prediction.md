---
title: "Lab: Insurance Claim Prediction"
chapter: false
weight: 130 
---

This section walks you through training your first model using Amazon SageMaker. You use notebooks and training algorithms provided by Amazon SageMaker.

This notebook describes using machine learning (ML) for Insurance Claim Prediction. We use the motor third-part liability policies dataset: <https://www.openml.org/d/41214>

The method that we'll use is xgboost. The XGBoost (eXtreme Gradient Boosting) is a popular and efficient open-source implementation of the gradient boosted trees algorithm. Gradient boosting is a supervised learning algorithm that attempts to accurately predict a target variable by combining an ensemble of estimates from a set of simpler and weaker models. The XGBoost algorithm performs well in machine learning competitions because of its robust handling of a variety of data types, relationships, distributions, and the variety of hyperparameters that you can fine-tune. You can use XGBoost for regression, classification (binary and multiclass), and ranking problems.

1. Access the Jupyter lab started earlier.
2. You need to specify the S3 bucket and prefix that you want to use for training and model data. This should be within the same region as the Notebook Instance, training, and hosting.

    * Create a bucket for this notebook (ie: "insurance-CURRENTDATE")
        * You can do that in the terminal of the Jupyter Notebook with this:

        ```bash
        aws s3api create-bucket --bucket insurance-`date +%s`
        ```

        * Copy the bucket name for future references.
3. On the left sidebar, navigate into `FSI/training/built-in-algorithms` , double click on `insurance-training-xgboost.ipynb` to open it.
4. You are now ready to begin the notebook.
5. In the notebook, set the bucket name. Change the bucket name to some of your own in the notebook and from then it can run without modifications.

    > To train the model in this notebook, you will use the `Amazon SageMaker Python SDK` which abstracts several implementation details, and is easy to use.


    ```python
    from sagemaker.estimator import Estimator
    training_model = Estimator(training_image,
                           role, 
                           instance_count=1, 
                           instance_type='ml.m5.2xlarge',
                           volume_size = 5,
                           max_run = 3600,
                           input_mode= 'File',
                           output_path=training_output_path,
                           sagemaker_session=feature_store_session)
    ``` 

6. Run each cell of the notebook.
