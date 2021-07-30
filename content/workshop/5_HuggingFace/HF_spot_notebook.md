---
title: "Huggingface Sagemaker-sdk - Spot instances"
chapter: false
weight: 20
---

### Binary Classification with `Trainer` and `imdb` dataset

1. [Introduction](#Introduction)  
2. [Development Environment and Permissions](#Development-Environment-and-Permissions)
    1. [Installation](#Installation)  
    2. [Development environment](#Development-environment)  
    3. [Permissions](#Permissions)
3. [Processing](#Preprocessing)   
    1. [Tokenization](#Tokenization)  
    2. [Uploading data to sagemaker_session_bucket](#Uploading-data-to-sagemaker_session_bucket)  
4. [Fine-tuning & starting Sagemaker Training Job](#Fine-tuning-\&-starting-Sagemaker-Training-Job)  
    1. [Creating an Estimator and start a training job](#Creating-an-Estimator-and-start-a-training-job)  
    2. [Estimator Parameters](#Estimator-Parameters)   
    3. [Download fine-tuned model from s3](#Download-fine-tuned-model-from-s3)
    3. [Attach to old training job to an estimator ](#Attach-to-old-training-job-to-an-estimator)  
5. [_Coming soon_:Push model to the Hugging Face hub](#Push-model-to-the-Hugging-Face-hub)

# Introduction

Welcome to our end-to-end binary Text-Classification example. In this demo, we will use the Hugging Faces `transformers` and `datasets` library together with a custom Amazon sagemaker-sdk extension to fine-tune a pre-trained transformer on binary text classification. In particular, the pre-trained model will be fine-tuned using the `imdb` dataset. To get started, we need to set up the environment with a few prerequisite steps, for permissions, configurations, and so on. This demo will also show you can use spot instances and continue training.

![image.png](attachment:image.png)

_**NOTE: You can run this demo in Sagemaker Studio, your local machine or Sagemaker Notebook Instances**_

# Development Environment and Permissions 

## Installation

_*Note:* we only install the required libraries from Hugging Face and AWS. You also need PyTorch or Tensorflow, if you haven´t it installed_


```python
!pip install "sagemaker>=2.48.0" "transformers==4.6.1" "datasets[s3]==1.6.2" --upgrade
```

## Development environment 

**upgrade ipywidgets for `datasets` library and restart kernel, only needed when prerpocessing is done in the notebook**


```python
%%capture
import IPython
!conda install -c conda-forge ipywidgets -y
IPython.Application.instance().kernel.do_shutdown(True) # has to restart kernel so changes are used
```


```python
import sagemaker.huggingface
```

## Permissions

_If you are going to use Sagemaker in a local environment. You need access to an IAM Role with the required permissions for Sagemaker. You can find [here](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html) more about it._


```python
import sagemaker

sess = sagemaker.Session()
# sagemaker session bucket -> used for uploading data, models and logs
# sagemaker will automatically create this bucket if it not exists
sagemaker_session_bucket=None
if sagemaker_session_bucket is None and sess is not None:
    # set to default bucket if a bucket name is not given
    sagemaker_session_bucket = sess.default_bucket()

role = sagemaker.get_execution_role()
sess = sagemaker.Session(default_bucket=sagemaker_session_bucket)

print(f"sagemaker role arn: {role}")
print(f"sagemaker bucket: {sess.default_bucket()}")
print(f"sagemaker session region: {sess.boto_region_name}")
```

# Preprocessing

We are using the `datasets` library to download and preprocess the `imdb` dataset. After preprocessing, the dataset will be uploaded to our `sagemaker_session_bucket` to be used within our training job. The [imdb](http://ai.stanford.edu/~amaas/data/sentiment/) dataset consists of 25000 training and 25000 testing highly polar movie reviews.

## Tokenization 


```python
from datasets import load_dataset
from transformers import AutoTokenizer

# tokenizer used in preprocessing
tokenizer_name = 'distilbert-base-uncased'

# dataset used
dataset_name = 'imdb'

# s3 key prefix for the data
s3_prefix = 'samples/datasets/imdb'
```


```python
# load dataset
dataset = load_dataset(dataset_name)

# download tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# tokenizer helper function
def tokenize(batch):
    return tokenizer(batch['text'], padding='max_length', truncation=True)

# load dataset
train_dataset, test_dataset = load_dataset('imdb', split=['train', 'test'])
test_dataset = test_dataset.shuffle().select(range(10000)) # smaller the size for test dataset to 10k 


# tokenize dataset
train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

# set format for pytorch
train_dataset =  train_dataset.rename_column("label", "labels")
train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset = test_dataset.rename_column("label", "labels")
test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
```

## Uploading data to `sagemaker_session_bucket`

After we processed the `datasets` we are going to use the new `FileSystem` [integration](https://huggingface.co/docs/datasets/filesystems.html) to upload our dataset to S3.


```python
import botocore
from datasets.filesystems import S3FileSystem

s3 = S3FileSystem()  

# save train_dataset to s3
training_input_path = f's3://{sess.default_bucket()}/{s3_prefix}/train'
train_dataset.save_to_disk(training_input_path,fs=s3)

# save test_dataset to s3
test_input_path = f's3://{sess.default_bucket()}/{s3_prefix}/test'
test_dataset.save_to_disk(test_input_path,fs=s3)

```


```python
training_input_path = f's3://{sess.default_bucket()}/{s3_prefix}/train'
test_input_path = f's3://{sess.default_bucket()}/{s3_prefix}/test'

```

# Fine-tuning & starting Sagemaker Training Job

In order to create a sagemaker training job we need an `HuggingFace` Estimator. The Estimator handles end-to-end Amazon SageMaker training and deployment tasks. In a Estimator we define, which fine-tuning script should be used as `entry_point`, which `instance_type` should be used, which `hyperparameters` are passed in .....



```python
huggingface_estimator = HuggingFace(entry_point='train.py',
                            source_dir='./scripts',
                            base_job_name='huggingface-sdk-extension',
                            instance_type='ml.p3.2xlarge',
                            instance_count=1,
                            transformers_version='4.4',
                            pytorch_version='1.6',
                            py_version='py36',
                            role=role,
                            hyperparameters = {'epochs': 1,
                                               'train_batch_size': 32,
                                               'model_name':'distilbert-base-uncased'
                                                })
```

When we create a SageMaker training job, SageMaker takes care of starting and managing all the required ec2 instances for us with the `huggingface` container, uploads the provided fine-tuning script `train.py` and downloads the data from our `sagemaker_session_bucket` into the container at `/opt/ml/input/data`. Then, it starts the training job by running. 

```python
/opt/conda/bin/python train.py --epochs 1 --model_name distilbert-base-uncased --train_batch_size 32
```

The `hyperparameters` you define in the `HuggingFace` estimator are passed in as named arguments. 

Sagemaker is providing useful properties about the training environment through various environment variables, including the following:

* `SM_MODEL_DIR`: A string that represents the path where the training job writes the model artifacts to. After training, artifacts in this directory are uploaded to S3 for model hosting.

* `SM_NUM_GPUS`: An integer representing the number of GPUs available to the host.

* `SM_CHANNEL_XXXX:` A string that represents the path to the directory that contains the input data for the specified channel. For example, if you specify two input channels in the HuggingFace estimator’s fit call, named `train` and `test`, the environment variables `SM_CHANNEL_TRAIN` and `SM_CHANNEL_TEST` are set.


To run your training job locally you can define `instance_type='local'` or `instance_type='local-gpu'` for gpu usage. _Note: this does not working within SageMaker Studio_



```python
!pygmentize ./scripts/train.py
```

    [34mfrom[39;49;00m [04m[36mtransformers[39;49;00m [34mimport[39;49;00m AutoModelForSequenceClassification, Trainer, TrainingArguments
    [34mfrom[39;49;00m [04m[36mtransformers[39;49;00m[04m[36m.[39;49;00m[04m[36mtrainer_utils[39;49;00m [34mimport[39;49;00m get_last_checkpoint
    
    [34mfrom[39;49;00m [04m[36msklearn[39;49;00m[04m[36m.[39;49;00m[04m[36mmetrics[39;49;00m [34mimport[39;49;00m accuracy_score, precision_recall_fscore_support
    [34mfrom[39;49;00m [04m[36mdatasets[39;49;00m [34mimport[39;49;00m load_from_disk
    [34mimport[39;49;00m [04m[36mlogging[39;49;00m
    [34mimport[39;49;00m [04m[36msys[39;49;00m
    [34mimport[39;49;00m [04m[36margparse[39;49;00m
    [34mimport[39;49;00m [04m[36mos[39;49;00m
    
    [37m# Set up logging[39;49;00m
    logger = logging.getLogger([31m__name__[39;49;00m)
    
    logging.basicConfig(
        level=logging.getLevelName([33m"[39;49;00m[33mINFO[39;49;00m[33m"[39;49;00m),
        handlers=[logging.StreamHandler(sys.stdout)],
        [36mformat[39;49;00m=[33m"[39;49;00m[33m%(asctime)s[39;49;00m[33m - [39;49;00m[33m%(name)s[39;49;00m[33m - [39;49;00m[33m%(levelname)s[39;49;00m[33m - [39;49;00m[33m%(message)s[39;49;00m[33m"[39;49;00m,
    )
    
    [34mif[39;49;00m [31m__name__[39;49;00m == [33m"[39;49;00m[33m__main__[39;49;00m[33m"[39;49;00m:
    
        logger.info(sys.argv)
    
        parser = argparse.ArgumentParser()
    
        [37m# hyperparameters sent by the client are passed as command-line arguments to the script.[39;49;00m
        parser.add_argument([33m"[39;49;00m[33m--epochs[39;49;00m[33m"[39;49;00m, [36mtype[39;49;00m=[36mint[39;49;00m, default=[34m3[39;49;00m)
        parser.add_argument([33m"[39;49;00m[33m--train-batch-size[39;49;00m[33m"[39;49;00m, [36mtype[39;49;00m=[36mint[39;49;00m, default=[34m32[39;49;00m)
        parser.add_argument([33m"[39;49;00m[33m--eval-batch-size[39;49;00m[33m"[39;49;00m, [36mtype[39;49;00m=[36mint[39;49;00m, default=[34m64[39;49;00m)
        parser.add_argument([33m"[39;49;00m[33m--warmup_steps[39;49;00m[33m"[39;49;00m, [36mtype[39;49;00m=[36mint[39;49;00m, default=[34m500[39;49;00m)
        parser.add_argument([33m"[39;49;00m[33m--model_name[39;49;00m[33m"[39;49;00m, [36mtype[39;49;00m=[36mstr[39;49;00m)
        parser.add_argument([33m"[39;49;00m[33m--learning_rate[39;49;00m[33m"[39;49;00m, [36mtype[39;49;00m=[36mstr[39;49;00m, default=[34m5e-5[39;49;00m)
        parser.add_argument([33m"[39;49;00m[33m--output_dir[39;49;00m[33m"[39;49;00m, [36mtype[39;49;00m=[36mstr[39;49;00m)
    
        [37m# Data, model, and output directories[39;49;00m
        parser.add_argument([33m"[39;49;00m[33m--output-data-dir[39;49;00m[33m"[39;49;00m, [36mtype[39;49;00m=[36mstr[39;49;00m, default=os.environ[[33m"[39;49;00m[33mSM_OUTPUT_DATA_DIR[39;49;00m[33m"[39;49;00m])
        parser.add_argument([33m"[39;49;00m[33m--model-dir[39;49;00m[33m"[39;49;00m, [36mtype[39;49;00m=[36mstr[39;49;00m, default=os.environ[[33m"[39;49;00m[33mSM_MODEL_DIR[39;49;00m[33m"[39;49;00m])
        parser.add_argument([33m"[39;49;00m[33m--n_gpus[39;49;00m[33m"[39;49;00m, [36mtype[39;49;00m=[36mstr[39;49;00m, default=os.environ[[33m"[39;49;00m[33mSM_NUM_GPUS[39;49;00m[33m"[39;49;00m])
        parser.add_argument([33m"[39;49;00m[33m--training_dir[39;49;00m[33m"[39;49;00m, [36mtype[39;49;00m=[36mstr[39;49;00m, default=os.environ[[33m"[39;49;00m[33mSM_CHANNEL_TRAIN[39;49;00m[33m"[39;49;00m])
        parser.add_argument([33m"[39;49;00m[33m--test_dir[39;49;00m[33m"[39;49;00m, [36mtype[39;49;00m=[36mstr[39;49;00m, default=os.environ[[33m"[39;49;00m[33mSM_CHANNEL_TEST[39;49;00m[33m"[39;49;00m])
    
        args, _ = parser.parse_known_args()
    
        [37m# load datasets[39;49;00m
        train_dataset = load_from_disk(args.training_dir)
        test_dataset = load_from_disk(args.test_dir)
    
        logger.info([33mf[39;49;00m[33m"[39;49;00m[33m loaded train_dataset length is: [39;49;00m[33m{[39;49;00m[36mlen[39;49;00m(train_dataset)[33m}[39;49;00m[33m"[39;49;00m)
        logger.info([33mf[39;49;00m[33m"[39;49;00m[33m loaded test_dataset length is: [39;49;00m[33m{[39;49;00m[36mlen[39;49;00m(test_dataset)[33m}[39;49;00m[33m"[39;49;00m)
    
        [37m# compute metrics function for binary classification[39;49;00m
        [34mdef[39;49;00m [32mcompute_metrics[39;49;00m(pred):
            labels = pred.label_ids
            preds = pred.predictions.argmax(-[34m1[39;49;00m)
            precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=[33m"[39;49;00m[33mbinary[39;49;00m[33m"[39;49;00m)
            acc = accuracy_score(labels, preds)
            [34mreturn[39;49;00m {[33m"[39;49;00m[33maccuracy[39;49;00m[33m"[39;49;00m: acc, [33m"[39;49;00m[33mf1[39;49;00m[33m"[39;49;00m: f1, [33m"[39;49;00m[33mprecision[39;49;00m[33m"[39;49;00m: precision, [33m"[39;49;00m[33mrecall[39;49;00m[33m"[39;49;00m: recall}
    
        [37m# download model from model hub[39;49;00m
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    
        [37m# define training args[39;49;00m
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            warmup_steps=args.warmup_steps,
            evaluation_strategy=[33m"[39;49;00m[33mepoch[39;49;00m[33m"[39;49;00m,
            logging_dir=[33mf[39;49;00m[33m"[39;49;00m[33m{[39;49;00margs.output_data_dir[33m}[39;49;00m[33m/logs[39;49;00m[33m"[39;49;00m,
            learning_rate=[36mfloat[39;49;00m(args.learning_rate),
        )
    
        [37m# create Trainer instance[39;49;00m
        trainer = Trainer(
            model=model,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
        )
    
        [37m# train model[39;49;00m
        [34mif[39;49;00m get_last_checkpoint(args.output_dir) [35mis[39;49;00m [35mnot[39;49;00m [34mNone[39;49;00m:
            logger.info([33m"[39;49;00m[33m***** continue training *****[39;49;00m[33m"[39;49;00m)
            trainer.train(resume_from_checkpoint=args.output_dir)
        [34melse[39;49;00m:
            trainer.train()
        [37m# evaluate model[39;49;00m
        eval_result = trainer.evaluate(eval_dataset=test_dataset)
    
        [37m# writes eval result to file which can be accessed later in s3 ouput[39;49;00m
        [34mwith[39;49;00m [36mopen[39;49;00m(os.path.join(args.output_data_dir, [33m"[39;49;00m[33meval_results.txt[39;49;00m[33m"[39;49;00m), [33m"[39;49;00m[33mw[39;49;00m[33m"[39;49;00m) [34mas[39;49;00m writer:
            [36mprint[39;49;00m([33mf[39;49;00m[33m"[39;49;00m[33m***** Eval results *****[39;49;00m[33m"[39;49;00m)
            [34mfor[39;49;00m key, value [35min[39;49;00m [36msorted[39;49;00m(eval_result.items()):
                writer.write([33mf[39;49;00m[33m"[39;49;00m[33m{[39;49;00mkey[33m}[39;49;00m[33m = [39;49;00m[33m{[39;49;00mvalue[33m}[39;49;00m[33m\n[39;49;00m[33m"[39;49;00m)
    
        [37m# Saves the model to s3[39;49;00m
        trainer.save_model(args.model_dir)


## Creating an Estimator and start a training job


```python
from sagemaker.huggingface import HuggingFace

# hyperparameters, which are passed into the training job
hyperparameters={'epochs': 1,
                 'train_batch_size': 32,
                 'model_name':'distilbert-base-uncased',
                 'output_dir':'/opt/ml/checkpoints'
                 }

# s3 uri where our checkpoints will be uploaded during training
job_name = "using-spot"
checkpoint_s3_uri = f's3://{sess.default_bucket()}/{job_name}/checkpoints'
```


```python
huggingface_estimator = HuggingFace(entry_point='train.py',
                            source_dir='./scripts',
                            instance_type='ml.p3.2xlarge',
                            instance_count=1,
                            base_job_name=job_name,
                            checkpoint_s3_uri=checkpoint_s3_uri,
                            use_spot_instances=True,
                            max_wait=3600, # This should be equal to or greater than max_run in seconds'
                            max_run=1000, # expected max run in seconds
                            role=role,
                            transformers_version='4.6',
                            pytorch_version='1.7',
                            py_version='py36',
                            hyperparameters = hyperparameters)
```


```python
# starting the train job with our uploaded datasets as input
huggingface_estimator.fit({'train': training_input_path, 'test': test_input_path})

# Training seconds: 874
# Billable seconds: 262
# Managed Spot Training savings: 70.0%
```

## Deploying the endpoint

To deploy our endpoint, we call `deploy()` on our HuggingFace estimator object, passing in our desired number of instances and instance type.


```python
predictor = huggingface_estimator.deploy(1,"ml.g4dn.xlarge")
```

Then, we use the returned predictor object to call the endpoint.


```python
sentiment_input= {"inputs":"I love using the new Inference DLC."}

predictor.predict(sentiment_input)
```

Finally, we delete the endpoint again.


```python
predictor.delete_endpoint()
```

# Extras

## Estimator Parameters


```python
# container image used for training job
print(f"container image used for training job: \n{huggingface_estimator.image_uri}\n")

# s3 uri where the trained model is located
print(f"s3 uri where the trained model is located: \n{huggingface_estimator.model_data}\n")

# latest training job name for this estimator
print(f"latest training job name for this estimator: \n{huggingface_estimator.latest_training_job.name}\n")


```

    container image used for training job: 
    558105141721.dkr.ecr.us-east-1.amazonaws.com/huggingface-training:pytorch1.6.0-transformers4.2.2-tokenizers0.9.4-datasets1.2.1-py36-gpu-cu110
    
    s3 uri where the trained model is located: 
    s3://philipps-sagemaker-bucket-us-east-1/huggingface-training-2021-02-04-16-47-39-189/output/model.tar.gz
    
    latest training job name for this estimator: 
    huggingface-training-2021-02-04-16-47-39-189
    



```python
# access the logs of the training job
huggingface_estimator.sagemaker_session.logs_for_job(huggingface_estimator.latest_training_job.name)
```

## Attach to old training job to an estimator 

In Sagemaker you can attach an old training job to an estimator to continue training, get results etc..


```python
from sagemaker.estimator import Estimator

# job which is going to be attached to the estimator
old_training_job_name=''
```


```python
# attach old training job
huggingface_estimator_loaded = Estimator.attach(old_training_job_name)

# get model output s3 from training job
huggingface_estimator_loaded.model_data
```

    
    2021-01-15 19:31:50 Starting - Preparing the instances for training
    2021-01-15 19:31:50 Downloading - Downloading input data
    2021-01-15 19:31:50 Training - Training image download completed. Training in progress.
    2021-01-15 19:31:50 Uploading - Uploading generated training model
    2021-01-15 19:31:50 Completed - Training job completed





    's3://philipps-sagemaker-bucket-eu-central-1/huggingface-sdk-extension-2021-01-15-19-14-13-725/output/model.tar.gz'


