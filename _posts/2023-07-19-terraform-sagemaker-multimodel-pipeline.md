---
title: 'Terraform a Scalable Comprehensive Sagemaker MultiModel Pipeline'
date: 2023-07-19
permalink: /posts/2023/07/terraform-sagemaker-multimodel-pipeline/
excerpt: "An end-to-end comprehensive scalable SageMaker pipeline for training, deploying and monitoring multiple models trained on different datasets to a single endpoint."
tags:
  - Sagemaker
  - MLOps
  - Hyperparameter Tuning
  - SKLearn
  - Multi-model Endpoint
  - Model Registry
  - Terraform
---
## Table of Contents
* [Architecture](#architecture)
  - [Modeling Pipeline](#modeling-pipeline)
  - [Deployment Pipeline](#deployment-pipeline)
* [Implementation](#implementation)
* [Extending Prebuilt Sagemaker SKLearn Image](#extending-prebuilt-sagemaker-sklearn-image)
* [ML Modeling Pipeline](#ml-modeling-pipeline)
  - [Pipeline Definition](#pipeline-definition)
  - [Preprocessing Step](#preprocessing-step)
  - [Hyperparameter Tuning With CrossValidation Step](#hyperparameter-tuning-with-crossvalidation-step)
  - [Refit Best Model:](#refit-best-model)
  - [Evaluation Step](#evaluation-step)
  - [Model Registry Step](#model-registry-step)
* [Deployment Pipeline](#deployment-pipeline-1)
  - [Deployer Lambda Function](#deployer-lambda-function)
  - [Invoker Lambda Function](#invoker-lambda-function)
  - [Multimodel Endpoint](#multimodel-endpoint)
* [Terraform The Pipeline](#terraform-the-pipeline)

In the ever-growing realm of machine learning, managing complex workflows efficiently is a crucial aspect. One such project that I recently did is a comprehensive, scalable SageMaker pipeline designed for training, deploying, and monitoring multiple models on varying datasets to a single endpoint. Built on the robust Amazon SageMaker platform, this project offers an end-to-end solution for defining and managing machine learning workflows. It covers everything from data preprocessing, model training, tuning, to the final deployment phase. 

The beauty of this pipeline lies in its scalability and cost-effectiveness. It employs multi-model endpoints, which use a shared serving container and the same fleet of resources to host all models. This approach significantly cuts down hosting costs by improving endpoint utilization compared to using single-model endpoints. The pipeline is meticulously designed to tune, train, and deploy multiple models on individual datasets, providing users with the flexibility to work with the data and infrastructures they prefer.

Another aspect of this project is that all the resources involved in implementing the pipeline are orchestrated using a `Terraform` script. In the upcoming sections of this blog post, we will delve deeper into the implementation details of this project exploring its architecture, codes and resources. Buckle up to gain a deeper understanding of this fascinating SageMaker pipeline project!

## Architecture
Now that we have a high-level understanding of the project, let's dive deeper into the specific architecture of the pipeline. The detailed architecture diagram, presented next, provides a visual representation of the entire system, showcasing the relationships between different components and steps involved in the pipeline. This diagram will help us comprehend the structure of the system and the sequential order of events in the pipeline.
![](https://github.com/sagar-spkt/sagemaker-e2e-pipeline/raw/main/docs/images/sagemaker-multimodel-pipeline.png)

The pipeline of this project is divided into two sections: the Modeling Pipeline and the Deployment Pipeline. Let's delve deeper into the steps involved in each of these sections:

### Modeling Pipeline
The Modeling Pipeline is the initial phase where the model is trained and prepared for deployment. Here's what happens in each step:

- **Preprocessing**: The raw data is cleaned and transformed into a suitable format for model training. This step is crucial as quality data is a prerequisite for training an effective model. It involves handling missing values, removing outliers, encoding categorical variables, and so on.

- **Hyperparameter Tuning**: This step involves searching for the best hyperparameters for the model. Hyperparameters are the configuration variables that govern the training process of a model. For instance, the learning rate in a neural network is a hyperparameter. A hyperparameter tuning algorithm, like grid search or random search, is used to explore different combinations of hyperparameters to find the optimal set that minimizes the loss function.

- **Refit Best Model**: After the best hyperparameters are found, the model is trained again using these hyperparameters. This step ensures that the model is the best version of itself before it is evaluated and potentially deployed.

- **Evaluate Best Model**: The performance of the best model is evaluated in this step. This is done using a holdout validation set that the model has never seen before. Evaluation metrics like accuracy, precision, recall, or AUC-ROC (for classification tasks), or MSE, MAE, R2 score (for regression tasks) are computed.

- **Registration Metric Check**: The model's performance metrics are checked against a predefined threshold or previous models' performance to decide whether to register the model in the registry. This step ensures that only models that meet the quality standards are registered for deployment.

- **Model Package Registration Step**: If the model passes the registration metric check, it is registered to the SageMaker Model Registry. This registry serves as a repository where trained models are stored before they are deployed.

### Deployment Pipeline
The Deployment Pipeline is the second phase where the registered models are deployed for serving predictions.

- The pipeline listens to approval/rejection events in the SageMaker Model Registry via AWS EventBridge. An approval event triggers the deployment of the approved model.

- The approved models are deployed to an endpoint's multi-model artifact location in S3 using a Lambda function. AWS Lambda is a serverless compute service that lets you run your code without provisioning or managing servers.

- Another AWS Lambda function with a Function URL is used to interact with the SageMaker endpoint. This function can be used to send data to the endpoint for inference and receive predictions.

- The scalability of the SageMaker endpoint is managed by AWS Application Auto Scaling. This service can automatically adjust the capacity of the endpoint to maintain steady, predictable performance at the lowest possible cost.

Overall, these steps ensure a streamlined process from data preprocessing to model deployment, providing an efficient and scalable solution for machine learning workflows.

## Implementation
Now that we've understood the architecture of the pipeline, it's time to delve into the implementation details. In the following section, we will walk through the process of implementing the pipeline to train separate models for two different datasets: the Breast Cancer dataset and the Bank Note Authentication dataset. The pipeline will do a hyperparameter search across multiple SKLearn, XGBoost and LightGBM classifiers and deploys the best-found model for each dataset in a single scalable multimodel Sagemaker endpoint. You'll find that you can adjust the scripts to include more datasets and custom preprocessing and training steps as you go through the blog. Our implementation will ensure that all steps from beginning to the end in the pipeline are scalable.

In this blog, I'll describe the main code snippets and components of the implementation. The complete implementation can be found in the following repo. This post is just a complementary to the Github Repo. Please read the [README](https://github.com/sagar-spkt/sagemaker-e2e-pipeline) and [USAGE_GUIDELINE](https://github.com/sagar-spkt/sagemaker-e2e-pipeline/blob/main/docs/USAGE_GUIDELINE.md).

[![](/images/blogs/github-sagemaker-e2e-multimodel-pipeline.jpg)](https://github.com/sagar-spkt/sagemaker-e2e-pipeline)

## Extending Prebuilt Sagemaker SKLearn Image
Docker images are the real backbone of the Sagemaker; whatever processing we do in each step of pipeline, we do it inside the container. In this section, we will discuss how to extend a prebuilt Sagemaker SKLearn Image. The main purpose of this is to add additional libraries that are not included in the prebuilt image, in this case, LightGBM and XGBoost. You can create custom Docker image from scratch if you want.

The Dockerfile for extending the SageMaker SKLearn Image can be written as follows:
<style>.emgithub-file .file-data {max-height: 500px;overflow-y: auto; scrollbar-width: thin;}</style>
<script src="https://emgithub.com/embed-v2.js?target=https%3A%2F%2Fgithub.com%2Fsagar-spkt%2Fsagemaker-e2e-pipeline%2Fblob%2Fmain%2Fimage%2FDockerfile&style=androidstudio&type=code&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on"></script>

From this Dockerfile, we then create docker image and push it to ecr using the following script. The script requires aws account id, aws region, and image name passed to it while running. Don't worry about running this script for now. We run this script using Terraform.

<script src="https://emgithub.com/embed-v2.js?target=https%3A%2F%2Fgithub.com%2Fsagar-spkt%2Fsagemaker-e2e-pipeline%2Fblob%2Fmain%2Fimage%2Fbuild_and_push.sh&style=androidstudio&type=code&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on"></script>

## ML Modeling Pipeline
Now that we have Docker container ready on which we can run our modeling jobs, let's create scripts for those jobs. Let's begin with creating a python script where we'll define the pipeline with Sagemaker Python SDK.

### Pipeline Definition:
We'll create a sagemaker pipeline definition python called `pipeline.py`. This file will define all the steps in [Modeling Pipeline](#modeling-pipeline). For now, let's import necessary dependencies, create arguments and sagemaker sessions necessary for the pipeline. Further down, we'll discuss the implementation of each step.
<script src="https://emgithub.com/embed-v2.js?target=https%3A%2F%2Fgithub.com%2Fsagar-spkt%2Fsagemaker-e2e-pipeline%2Fblob%2Fmain%2Fpipeline.py%23L1-L58&style=androidstudio&type=code&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on"></script>

### Preprocessing Step:
We'll prepare a python script for downloading, splitting, and saving two specific datasets: the Breast Cancer and the Banknote Authentication datasets.The main block of the script uses argparse to parse command line arguments for selecting the dataset, specifying the test size, random state, stratification, target column name and file name. It then calls the appropriate function to get the requested dataset.
Finally, the script creates directories for saving the training and testing datasets and saves these datasets as CSV files. You can modify the script to include other datasets and preprocessing steps. You just need to take care of `dataset` argument passed to the script and process that dataset only.
<script src="https://emgithub.com/embed-v2.js?target=https%3A%2F%2Fgithub.com%2Fsagar-spkt%2Fsagemaker-e2e-pipeline%2Fblob%2Fmain%2Fscripts%2Fpreprocessing.py&style=androidstudio&type=code&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on"></script>

We'll call this script from preprocessing step in the pipeline. In the `pipeline.py`, we define Preprocessing step using Sagemaker `ScriptProcessor` which runs the above script. We'll also define parameters that the user passes during the pipeline execution. Also, note that we have specified where the output artifacts like train/test split of the preprocessing steps should be dumped so that sagemaker will upload them to s3.
<script src="https://emgithub.com/embed-v2.js?target=https%3A%2F%2Fgithub.com%2Fsagar-spkt%2Fsagemaker-e2e-pipeline%2Fblob%2Fmain%2Fpipeline.py%23L60-L111&style=androidstudio&type=code&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on"></script>

### Hyperparameter Tuning With CrossValidation Step:
Hyperparameter tuning is basically running multiple training with different combination of model hyperparameters and selecting the model hyperparameter which gives the best result. When we do tuning with k fold cross-validation, we select models that has highest average performance across the folds. The tuning with cross-validation and training steps differs from each other in such a way that we look for peformance metric in tuning but train the model in whole set in training. We'll prepare script which can do both based on the cross-validation argument flag passed to it. Also, note that we're not looking for best hyperparameter of a single model in this pipeline. Models including RandomForest, Naive Bayes, NeuralNet, LogisticRegression, XGBoost, and LightGBM competes with each other during tuning. So for each models we create script that does cross-validation and training on whole dataset. For example, Following is the script for LightGBM:
<script src="https://emgithub.com/embed-v2.js?target=https%3A%2F%2Fgithub.com%2Fsagar-spkt%2Fsagemaker-e2e-pipeline%2Fblob%2Fmain%2Fscripts%2Festimators%2FLightGBM.py&style=androidstudio&type=code&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on"></script>

You can find script for other estimators in [**my github repo**](https://github.com/sagar-spkt/sagemaker-e2e-pipeline/tree/main/scripts/estimators).

Now the scripts are ready and later we'll upload them to s3 using terraform, we can work on implementing tuning step in the pipeline. First we'll create a dictionary of estimators where key is estimator name that match with the script name for training and value is sagemaker estimator. Along with the paramters required for the tuning steps, we define the hyperparameter search space for each estimators. Please have a look at the following code snippet.
<script src="https://emgithub.com/embed-v2.js?target=https%3A%2F%2Fgithub.com%2Fsagar-spkt%2Fsagemaker-e2e-pipeline%2Fblob%2Fmain%2Fpipeline.py%23L113-L257&style=androidstudio&type=code&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on"></script>

With those estimator and hyperparameters definition, we can create tuning step of the pipeline in the following way.
<script src="https://emgithub.com/embed-v2.js?target=https%3A%2F%2Fgithub.com%2Fsagar-spkt%2Fsagemaker-e2e-pipeline%2Fblob%2Fmain%2Fpipeline.py%23L259-L295&style=androidstudio&type=code&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on"></script>

### Refit Best Model:
After tuning job is completed, we need to retrain the model with the best found parameters on the whole training dataset. For that we'll create a script called `refit.py`. The Python script is essentially a command-line interface for launching AWS SageMaker training jobs. It extracts hyperparameters from specified best training job, disables cross-validation, and runs a Python script associated with a chosen algorithm using those hyperparameters. Have a look at the script.
<script src="https://emgithub.com/embed-v2.js?target=https%3A%2F%2Fgithub.com%2Fsagar-spkt%2Fsagemaker-e2e-pipeline%2Fblob%2Fmain%2Fscripts%2Frefit.py&style=androidstudio&type=code&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on"></script>

We can use this script to create best estimator in sagemaker and add Training Step in the pipeline which will act as Refit Best Model step in the pipeline.
<script src="https://emgithub.com/embed-v2.js?target=https%3A%2F%2Fgithub.com%2Fsagar-spkt%2Fsagemaker-e2e-pipeline%2Fblob%2Fmain%2Fpipeline.py%23L297-L322&style=androidstudio&type=code&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on"></script>

### Evaluation Step:
Now that we've trained the model in best found parameters, we can measure it's performance in test split we created in the preprocessing step. Let's create a evaluation script.The provided Python script is designed to evaluate the performance of a binary classification machine learning model. It begins by parsing command-line arguments for the test data file, the metric to register, the features, and the target feature. The script then extracts a model from a tarball file, loads the test data from a CSV file, and prepares the test data for evaluation. The model is used to make predictions on the test data and calculate several evaluation metrics, including precision, recall, accuracy, f1 score, Roc Auc score, and confusion matrix. Finally, a report including all the calculated metrics is generated and saved to a JSON file.
<script src="https://emgithub.com/embed-v2.js?target=https%3A%2F%2Fgithub.com%2Fsagar-spkt%2Fsagemaker-e2e-pipeline%2Fblob%2Fmain%2Fscripts%2Fevaluate.py&style=androidstudio&type=code&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on"></script>

Next step is to add evaluation step in the pipeline. Following is the code snippet for that. The snippet initiates by defining parameters for model registration in the registry, including the evaluation metric, the minimum threshold for the metric, and the approval status. It then sets up an evaluation step to assess the best model using a ScriptProcessor object, which runs a Python script with specified arguments, inputs, and outputs. The evaluation step is encapsulated in a `ProcessingStep`` object, which signifies this step in the SageMaker pipeline.
<script src="https://emgithub.com/embed-v2.js?target=https%3A%2F%2Fgithub.com%2Fsagar-spkt%2Fsagemaker-e2e-pipeline%2Fblob%2Fmain%2Fpipeline.py%23L324-L393&style=androidstudio&type=code&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on"></script>

### Model Registry Step:
After the evaluation is complete, if the model performance exceeds some predefined threshold, we can register the model in model registry. The follwing provided Python code snippet leverages the Amazon SageMaker Python SDK to create and register a machine learning model. Initially, it defines the model metrics and creates an instance of the model using the `SKLearnModel` class. The model is then registered using its `register` method, where important parameters like the content types for the request and response, instances for inference and transformation, and the model metrics are specified. A condition check is implemented to verify if the model's evaluation metric value meets the required threshold. Depending on the condition's outcome, the model registration step or a failure step is executed. This code provides a thorough procedure for setting up a machine learning model, registering it, and evaluating its performance.
<script src="https://emgithub.com/embed-v2.js?target=https%3A%2F%2Fgithub.com%2Fsagar-spkt%2Fsagemaker-e2e-pipeline%2Fblob%2Fmain%2Fpipeline.py%23L395-L467&style=androidstudio&type=code&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on"></script>

This concludes the modeling pipeline. Now, we'll delve into the deployment pipeline.

## Deployment Pipeline
The Deployment Pipeline is the phase where approved machine learning models, registered in the SageMaker Model Registry, are deployed for serving predictions. The pipeline uses AWS Lambda functions to deploy the models to an S3 location and to interact with the SageMaker endpoint for sending data and receiving predictions. AWS Application Auto Scaling manages the scalability of the endpoint, ensuring steady performance at the lowest cost. This process provides an efficient and scalable solution for machine learning workflows. Let's begin the pipeline defintion with the lambda function that handles the events in Sagemaker Model Registry like model approval or rejection and updates the model for each dataset in serving.

### Deployer Lambda Function:
The Python script below is designed to manage multiple machine learning models in the Amazon Web Services (AWS) environment, specifically using Amazon SageMaker and Amazon S3. It uses the Boto3 library, the Python SDK for AWS, to interact with these services. The script fetches the latest approved model package from a model group, stores a map of group names(dataset) to latest models in an S3 bucket, updates the model in endpoint model artifact S3 bucket, and handles EventBridge events that are triggered when the models are approved or rejected in the sagemaker model registry.
<script src="https://emgithub.com/embed-v2.js?target=https%3A%2F%2Fgithub.com%2Fsagar-spkt%2Fsagemaker-e2e-pipeline%2Fblob%2Fmain%2Flambda%2Fendpoint_deploy.py&style=androidstudio&type=code&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on"></script>

### Invoker Lambda Function:
The following Python script is an AWS Lambda function designed to interact with a SageMaker endpoint. It imports necessary libraries, sets up AWS clients, and defines two functions. The function `get_groupname2model_map` retrieves a JSON object from an S3 bucket mapping group names(dataset) to latest model names. The main function, `lambda_handler`, retrieves the group-to-model map, and checks if the specified model name exists in the map. If it does, it invokes the SageMaker endpoint with that model as target model, passing in the data from the event body, and returns the result with a 200 status code. If the model name doesn't exist, it returns a 404 status code with an error message. Essentially, this script serves as an interface between an HTTP request and a multiple machine learning model hosted on AWS SageMaker.
<script src="https://emgithub.com/embed-v2.js?target=https%3A%2F%2Fgithub.com%2Fsagar-spkt%2Fsagemaker-e2e-pipeline%2Fblob%2Fmain%2Flambda%2Fendpoint_invoke.py&style=androidstudio&type=code&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on"></script>

### Multimodel Endpoint:
We'll need a sagemaker endpoint along with model and endpoint configuration in multi-model mode. These resources will be created from Terraform which is discussed in the following section.

## Terraform The Pipeline
We'll create all the resources including AWS Infrastructure, Docker Images, Pipeline Definition, etc. using the following terraform file.
<script src="https://emgithub.com/embed-v2.js?target=https%3A%2F%2Fgithub.com%2Fsagar-spkt%2Fsagemaker-e2e-pipeline%2Fblob%2Fmain%2Fmain.tf&style=androidstudio&type=code&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on"></script>
Here is the description for each resources used in this Terraform.
- First, we define the required providers and variables for the pipeline.
- Next, we create an AWS IAM Role with access to necessary resources.
- A ECR repository is then created to host the custom images used in the pipeline.
- Since, terraform doesn't have inbuilt capacity to build an push docker images, we run the `build_and_push.sh` script we created at the beginning using Terraform's `null_resource`. 
- A S3 bucket for storing artifacts for and from the pipeline is created.
- tarball file of the source scripts is created, and uploaded to the bucket created above along with the separate preprocessing and evaluation script.
- The `pipeline.py` file is ran again with the Terraform's `null_resource`.
- The produced pipeline definition file is uploaded to the S3 bucket.
- A sagemaker pipeline is then created with the content of the pipeline definition file
- Next, a sagemaker endpoint model in `MultiModel` mode is created. It will points to a location is the S3 bucket where all the models for endpoint is stored. Our deployer lambda function will update the model in this location.
- A sagemaker endpoint configuration with attributes before scaling is created.
- The endpoint is than deployed with the configuration just created above.
- AWS App Auto Scaling is deployed with the endpoint as the target. The policy used is based on the CPU utilization of the instances in the sagemaker endpoint.
- The deployment lambda function is created.
- The EventBridge rule with the deployment lambda function as a target is created to handle the event the model registry.
- The invocation lambda function is created and function url pointing to the same lambda function is created.

We've come to the conclusion of the blog. Here, we created a scalable end-to-end ML Pipeline. For usage description, please refer to [my github repo](https://github.com/sagar-spkt/sagemaker-e2e-pipeline).