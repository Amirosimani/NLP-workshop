---
title: "Lab: Insurance Inference"
chapter: false
weight: 130 
---



1. Access the Jupyter lab started earlier.
2. You need to specify the S3 bucket and prefix that you want to use for training and model data. This should be within the same region as the Notebook Instance, training, and hosting.

    * Create a bucket for this notebook (ie: "insurance-CURRENTDATE")
        * You can do that in the terminal of the Jupyter Notebook with this:

        ```bash
        aws s3api create-bucket --bucket insurance-`date +%s`
        ```

        * Copy the bucket name for future references.
3. On the left sidebar, navigate into `FSI` , double click on `insurance-inference.ipynb` to open it.
4. You are now ready to begin the notebook.
5. In the notebook, set the bucket name. Change the bucket name to some of your own in the notebook and from then it can run without modifications.
6. Run each cell of the notebook.
