---
title: "Lab: Amazon Sagemaker Pipelines"
chapter: false
weight: 615
---

### Amazon SageMaker Pipelines

We can use Amazon Sagemaker Pipelines for end-to-end deployment of machine learning pipelines from pre-processing. training to deployment.  
1. Amazon Sagemaker pipelines makes it easy to create machine learning pipelines in studio to easily replicate and automate your sequence of steps
2. Amazon Sagemaker pipelines also makes it easy for operations teams to ensure only approved models are deployed in production with right level of automation

#### Lab

1. On launch tab click on new project

![Pipelines image](/images/pipeline-new-project.png)

    Existing pipelines can be accessed from left tab -> components and registeries 

![Pipelines image](/images/pipelines-existing-left-tab-.png)

2. After clicking on new project -> you will see three MLOps project templates to select from

    Click on MLops template for model building, training and deployment -> select Create project button at the bottom

![Pipelines image](/images/pipelines-create-newproject-from-template.png)

    New project creation will start at the background by Amazon sagemaker

3. Once New project creation completes, clone repo for both model deploy and model build

![Pipelines image](/images/pipelines-clone-repo-locally.png)

Cloned repo can be found in folder structure now by clicking on file brower on top left corner 

![Pipelines image](/images/pipelines-cloned-folders.png)

Within locally cloned repo, there should be two folders <br />
    - One with model build code <br />
    - Other with model deployment code


* __Model Build__ folder contains sample code used for automating model building workflow using Amazon Sagemaker Pipelines and can be used by data scientist, analyst and developers 
* __Model Deploy__ folder contains logic for model deployment to staging and production envirnoments using Cloudformation and can be used by MLOps teams


4. Lets Inspect Model build folder which contains __Amazon Sagemaker pipelines__ code for automating ML steps

Go sagemaker-project-name-modelbuild folder -> pipelines

![Pipelines image](/images/pipelines-build-folder-structure-1.png)
