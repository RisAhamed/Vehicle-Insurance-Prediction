# Vehicle Insurance Cross-Sell Prediction MLOps Pipeline

This project implements an end-to-end MLOps pipeline for predicting whether a customer would be interested in vehicle insurance. The pipeline includes data ingestion from MongoDB, data validation, transformation, model training, evaluation, and deployment to AWS S3.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)
- [Running the Pipeline](#running-the-pipeline)
- [Components](#components)
- [AWS Integration](#aws-integration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Overview

This MLOps pipeline automates the entire machine learning lifecycle for vehicle insurance cross-sell prediction:
1. Data ingestion from MongoDB or CSV files
2. Data validation to ensure data quality
3. Data transformation for feature engineering
4. Model training using various classification algorithms
5. Model evaluation against existing models
6. Model deployment to AWS S3 for production use

## Project Structure

```
Mlops-with-Mongo-DB/
├── app.py                  # Main application entry point
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (create this file)
├── src/
│   ├── components/         # Pipeline components
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   ├── model_evaluation.py
│   │   └── model_pusher.py
│   ├── cloud_storage/      # AWS S3 integration
│   │   └── aws_storage.py
│   ├── configuration/      # Configuration classes
│   │   └── aws_connection.py
│   ├── entity/             # Data classes
│   │   ├── artifact_entity.py
│   │   ├── config_entity.py
│   │   └── s3_estimator.py
│   ├── exception.py        # Custom exception handling
│   ├── logger.py           # Logging configuration
│   └── pipline/            # Pipeline orchestration
│       └── training_pipeline.py
└── artifact/               # Generated artifacts (created during runtime)
```

## Prerequisites

- Python 3.8 or higher
- AWS account with S3 access
- MongoDB account (optional, for data storage)
- Git

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/RisAhamed/Vehicle-Insurance-Prediction.git
cd Vehicle-Insurance-Prediction
```

### 2. Create a Virtual Environment

```bash
# For Windows
python -m venv vehicle
vehicle\Scripts\activate

# For macOS/Linux
python -m venv vehicle
source vehicle/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure AWS Credentials

Create a `.env` file in the root directory with the following content:

```
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=your_aws_region
S3_BUCKET_NAME=your_s3_bucket_name
```

Make sure your AWS user has permissions to access and write to S3 buckets.

### 5. Create S3 Bucket

Create an S3 bucket in your AWS account to store the models. The bucket name should match the `S3_BUCKET_NAME` in your `.env` file.

### 6. Configure MongoDB (Optional)

If you're using MongoDB for data storage, add the following to your `.env` file:

```
MONGODB_URL=your_mongodb_connection_string
DATABASE_NAME=your_database_name
COLLECTION_NAME=your_collection_name
```

## Running the Pipeline

To run the complete ML pipeline:

```bash
python app.py
```

This will:
1. Ingest data from the configured source
2. Validate the data structure and quality
3. Transform the data for model training
4. Train a classification model
5. Evaluate the model against any existing models in S3
6. Push the model to S3 if it performs better than existing models

## Components

### Data Ingestion

The data ingestion component can load data from:
- MongoDB collection
- Local CSV files
- Remote data sources

The ingested data is split into training and testing sets.

### Data Validation

Validates the ingested data by:
- Checking for required columns
- Validating data types
- Ensuring data completeness
- Generating validation reports

### Data Transformation

Transforms the raw data by:
- Handling categorical features
- Scaling numerical features
- Feature engineering
- Encoding target variables

### Model Trainer

Trains a classification model using:
- Various algorithms (Random Forest, XGBoost, etc.)
- Hyperparameter tuning
- Cross-validation
- Performance metrics calculation

### Model Evaluation

Evaluates the trained model by:
- Comparing with existing models in S3
- Calculating performance metrics (F1 score, precision, recall)
- Determining if the new model should replace the existing one

### Model Pusher

Pushes the model to production by:
- Saving the model to AWS S3
- Updating model metadata
- Versioning the model

## AWS Integration

This project uses AWS S3 for model storage and versioning:

1. **Model Storage**: Trained models are stored in S3 buckets
2. **Model Versioning**: Models are versioned based on training timestamps
3. **Model Retrieval**: The best model is retrieved for predictions

## Troubleshooting

### Common Issues

1. **S3 Access Issues**:
   - Ensure AWS credentials are correctly set in the `.env` file
   - Verify that the IAM user has appropriate S3 permissions

2. **First Run Issues**:
   - On the first run, no model exists in S3, so the pipeline will automatically push the newly trained model
   - Subsequent runs will compare new models with existing ones

3. **Data Validation Failures**:
   - Check the validation reports in the artifact directory
   - Ensure your data matches the expected schema

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request
