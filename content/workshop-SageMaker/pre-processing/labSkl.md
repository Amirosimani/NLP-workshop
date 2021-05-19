---
title: "Lab: SageMaker Processing (Optional)"
chapter: false
weight: 315 
---

### SageMaker Processing

This notebook shows how you can:

1. Run a processing job to run a scikit-learn script that cleans, pre-processes, performs feature engineering, and splits the input data into train and test sets.
2. Run a training job on the pre-processed training data to train a model
3. Run a processing job on the pre-processed test data to evaluate the trained model's performance
4. Use your own custom container to run processing jobs with your own Python libraries and dependencies.


Let's open our notebook and begin the lab:

1. Access the notebook instance you created earlier.
2. On the left sidebar, click the Folder icon and navigate into the `MLAI-workshop-resources/pre-processing/scikit_learn_processing_and_model_evaluation/` folder. Double-click on `scikit_learn_processing_and_model_evaluation.ipynb` to open it.
3. You are now ready to begin the notebook. Follow through the provided steps to build your custom container, and then leverage it for training in SageMaker.
