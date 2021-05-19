---
title: "Lab: House Price Prediction"
chapter: false
weight: 130 
---


This notebook describes using machine learning (ML) for predicting price of house. We use the Boston Housing dataset, present in Scikit-Learn: <https://scikit-learn.org/stable/datasets/index.html#boston-dataset.>

The method that we'll use is a linear regressor. Amazon SageMaker's __Linear Learner__ algorithm extends upon typical linear models by training many models in parallel, in a computationally efficient manner.

1. Access the Jupyter lab started earlier.
2. You need to specify the S3 bucket and prefix that you want to use for training and model data. This should be within the same region as the Notebook Instance, training, and hosting.

    * Create a bucket for this notebook (ie: "house-price-CURRENTDATE")
        * You can do that in the terminal of the Jupyter Notebook with this:

        ```bash
        aws s3api create-bucket --bucket house-price-`date +%s`
        ```

        * Copy the bucket name for future references.
3. On the left sidebar, navigate into `MLAI-workshop-resources/inference/built-in-algorithms` , double click on `linearLearner_boston_house.ipynb` to open it.
4. You are now ready to begin the notebook.
5. In the notebook, set the bucket name. Change the bucket name to some of your own in the notebook and from then it can run without modifications.
6. Run each cell of the notebook.
