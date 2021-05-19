---
title: "Lab: Population Segmentation"
chapter: false
weight: 120 
---

This section walks you through training your first model using Amazon SageMaker. You use notebooks and training algorithms provided by Amazon SageMaker.

This notebook describes using machine learning (ML) for population segmentation. We have taken publicly available, anonymized data from the US census on demographics by different US counties. The outcome of this analysis are natural groupings of similar counties in a transformed feature space.

There are two goals for this exercise:

1) Walk through a data science workflow using Amazon SageMaker for unsupervised learning using PCA and Kmeans modelling techniques.

2) Demonstrate how users can access the underlying models that are built within Amazon SageMaker to extract useful model attributes. Often, it can be difficult to draw conclusions from unsupervised learning, so being able to access the models for PCA and Kmeans becomes even more important beyond simply generating predictions using the model.

To start the lab:

1. Access the Jupyter lab started earlier.
2. You need to specify the S3 bucket and prefix that you want to use for training and model data. This should be within the same region as the Notebook Instance, training, and hosting.

    * Create a bucket for this notebook (ie: "clustering-CURRENTDATE")
        * You can do that in the terminal of the Jupyter Notebook with this:

        ```bash
        aws s3api create-bucket --bucket clustering-`date +%s`
        ```

        * Copy the bucket name for future references.
3. On the left sidebar, navigate into `MLAI-workshop-resources/training/built-in-algorithms` , double click on `sagemaker-clustering.ipynb` to open it.
4. You are now ready to begin the notebook.
5. In the notebook, set the bucket name. Change the bucket name to some of your own in the notebook and from then it can run without modifications.

    > To train the model in this notebook, you will use the `Amazon SageMaker Python SDK` which abstracts several implementation details, and is easy to use.


    ```python
    from sagemaker import KMeans

    num_clusters = 7
    kmeans = KMeans(role=role,
                    train_instance_count=1,
                    train_instance_type='ml.c4.xlarge',
                    output_path='s3://'+ bucket +'/counties/',
                    k=num_clusters)
    ``` 

6. Run each cell of the notebook.
