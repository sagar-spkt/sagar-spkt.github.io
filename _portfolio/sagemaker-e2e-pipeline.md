---
title: " Scalable End to End Sagemaker MultiModel Pipeline"
excerpt: "Worked on providing an end-to-end comprehensive scalable SageMaker pipeline for training and deploying multiple models trained on different datasets to a single endpoint."
collection: portfolio
date: 2023-07-15
comments: true
---

Overview
=====
The main goal of this project is to provide a reference implementation for building an end-to-end scalable multimodel pipeline with Amazon Sagemaker. Multi-model endpoints provide a scalable and cost-effective solution to deploying a large number of models trained on same or different datasets. They use the same fleet of resources and a shared serving container to host all of your models. This reduces hosting costs by improving endpoint utilization compared to using single-model endpoints. The pipeline is designed to tune, train and deploy multiple models on individual datasets the user wants to work with. It also provides a `Terraform` script that manages all the AWS resources involved in the pipeline. The deployment of AWS resources is automated through Github Actions. Refer to the following architecture diagram to understand the components involved and their relationship with each other in overall pipeline.

Architecture
=====
![](https://github.com/sagar-spkt/sagemaker-e2e-pipeline/raw/main/docs/images/sagemaker-multimodel-pipeline.png)
The pipeline is divided into two sections:
### Modeling Pipeline
This includes the following steps:

- Preprocessing: Cleaning and transforming raw data to suitable format for model training.
- Hyperparameter Tuning: Searching for the best hyperparameters for the model.
- Refit Best Model: Training the model with the best hyperparameters.
- Evaluate Best Model: Evaluating the performance of the best model.
- Registration Metric Check: Checking the model's performance metrics to decide whether to register model in registry.
- Model Package Registration Step: Registering the trained model to SageMaker Model Registry.

### Deployment Pipeline
This pipeline listens to Approval/rejection events in SageMaker Model Registry via EventBridge and deploys models to an endpoint's multimodel artifact location in S3 using a Lambda function. Another AWS Lambda with Function URL is used to interact with the Sagemaker endpoint, which is made scalable by Application Auto Scaling.

Github Repo
====
[![](/images/blogs/github-sagemaker-e2e-multimodel-pipeline.jpg)](https://github.com/sagar-spkt/sagemaker-e2e-pipeline)