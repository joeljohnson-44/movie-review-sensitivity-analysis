# Sentiment Analysis Project

## 1. Project Overview

This project implements a complete end-to-end sentiment analysis system using Machine Learning and FastAPI to classify text reviews into positive and negative classes.

This repo contains:
- Exploratory Data Analysis 
- Model building scripts
- Model training & evalution
- REST API scripts
- Docker container 
---

## 2. Dataset Information

We have used [IMDB-50k-Movie-Review-Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) containing labeled movie review text data for this project 

### 2.1 Dataset Structure

Each record contains:
- Text review/comment
- Corresponding sentiment label (positive/negative)

Example:

| Text                    | Label    |
|-------------------------|----------|
| I loved this movie      | Positive |
| Worst product ever      | Negative |

---
## 2.2 Exploratory Data Analysis
- No class Imbalance Present: Both positive and negative classes have equal count. 
- Number of words and sentences have no significant varation across classes. 
- Median count of sentences per review is about ~10 and count of words per review is ~200.  
For more details check the [exploratory_data_analysis.ipynb](src/exploratory_data_analysis.ipynb)

---

## 3. Model Development and Evaluation 

We considered NLTK model for benchmarking, then traditional ML models and transformer based BERT model. 

For the traditional Models (Logistic Regression, Random Forest, XGBoost) we used TF-IDF to convert reviews into numerical features, then trained the models on these vectors and for DistilBERT model we used pretrained weights and fed raw reviews directly; the model learned contextual features automatically. 

| Model | Details | Accuracy | F1-Score |
|-------|---------|---------|----------|
| 1 | Vader NLTK Sentiment Classifier | 70% | 74% |
| 2 | TF-IDF Vectorizer + Logistic Regression | 90% | 90% |
| 3 | TF-IDF Vectorizer + Random Forest | 87% | 87% |
| 4 | TF-IDF Vectorizer + XGBoost | 86% | 86% |
| 5 | DistilBERT (Using Pretrained Weights) | 94% | 94% |

*Key Observations*:

- DistilBERT achieved the highest performance with 94% accuracy and F1-score, indicating it can capture the nuances in text very effectively.

- TF-IDF + Logistic Regression also performed extremely well with 90% accuracy, while being computationally light, making it a strong candidate for scenarios with limited resources.

- Traditional ML models like Random Forest and XGBoost with TF-IDF-based performed slightly lower (86–87%), but still reasonably well.

- Vader NLTK, a rule-based sentiment analyzer, was the least accurate (70%)—useful for quick, lightweight analysis but less reliable for nuanced datasets.
---

## 4. API Development and Documentation

### 4.1 API Overview

The trained model is exposed as a REST API using:
- FastAPI
- Uvicorn
- Pydantic
- Swagger UI

API capabilities:
- Predict sentiment for single text
- Predict sentiment for batch texts
- Health check
- Model information

---
### 4.2 API Endpoint Documentation

#### [i] Home Endpoint

`GET /`

Response:

```json
{
  "message": "Sentiment Analysis API is running",
  "docs": "/docs",
  "health": "/health"
}
```

#### [ii] Single Prediction

`POST /predict`

Request Body:

```json
{
  "text": "I love this product"
}
```

Response:

```json
{
  "text": "I love this product",
  "sentiment": "positive"
}
```

#### [iii] Batch Prediction

`POST /predict_batch`

Request:

```json
{
  "texts": [
    "This is amazing",
    "Worst product ever"
  ]
}
```

Response:

```json
{
  "predictions": [
    {
      "text": "This is amazing",
      "sentiment": "positive"
    },
    {
      "text": "Worst product ever",
      "sentiment": "negative"
    }
  ]
}
```

#### [iv] Model Information

`GET /model_info`

---

## 4.3 Swagger Documentation

FastAPI automatically generates interactive documentation.

Open Swagger UI:
- `http://localhost:8000/docs`

Using Swagger, you can:
- View all endpoints
- Expand request schemas
- Test APIs directly
- See responses in real time

No external tool like Postman is required.

## 5. Setup 

### 5.1 Clone this repo  
```
git clone https://github.com/joeljohnson-44/movie-review-sensitivity-analysis.git 
```
After cloning move to this git repo.  

### 5.2 Download Models from Hugging Face
The model files are too large to store in Git, so I have added them in hugging face repo. 

- Go to the Hugging Face repository: https://huggingface.co/Joel-44/imdb50k-sensitivity-models/tree/main/distilbert_model
- Download the all the files in following folder: distilbert_model/
- Create a folder structure in your project:  
    movie-review-sensitivity-analysis/  
    └─ src/  
      └─ models/  
- Place the downloaded files in the src/models/distilbert_model folder:  
    movie-review-sensitivity-analysis/  
    └─ src/  
      └─ models/  
          └─ distilbert_model/  
- Verify the folder structure is correct:
  ls src/models 

### 5.3 Running the API Locally (Windows environment)

Step 1 — Install Dependencies

Please make sure python and pip are already installed in the syste, before this step. 
```bash
python -m venv venv
.venv\Scripts\Activate.ps1    
pip install -r requirements.txt
```

Step 2 — Start API

```bash
uvicorn app:app --reload
```

Step 3 — Open in Browser

- Home: `http://localhost:8000`
- Swagger UI: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/health`

### 5.4 Docker Deployment Documentation

#### Step 1: Building Docker Image

Please install docker in your environment before building docker images. 

Run from project root:

```bash
docker build -t sentiment-api .
```

#### Step 2: Running Docker Container

```bash
docker run -p 8000:8000 sentiment-api
```

#### Step 3:Verify Deployment

Open:
- `http://localhost:8000/docs`
- `http://localhost:8000/health`

API is now running inside Docker.

#### Step 4: Testing API Commands

### Testing with cURL

Single Prediction:

```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d "{\"text\":\"Great service\"}"
```

Batch Prediction:

```bash
curl -X POST "http://localhost:8000/predict_batch" -H "Content-Type: application/json" -d "{\"texts\":[\"good\",\"bad\",\"average\"]}"
```

---
## Additional Discussion

### Detect Model Degradation and Redeploy

To keep our model accurate in production, we implement continuous monitoring, testing, and automatic retraining using MLOps tools like MLflow and Seldon Core.

#### 1. Detecting Degradation

- (1.1) Batch Annotation: For each new batch of incoming data, we annotate a small representative portion to create a test set.

- (1.2) Performance Monitoring: The model predicts on this test set, and metrics like accuracy, F1-score, or RMSE are logged in MLflow.

- (1.3) Drift Detection: We monitor input feature distributions for drift (e.g., using Population Stability Index) and trigger alerts if metrics drop below thresholds.

*Monitoring Flow*
```
New Data Batch → Annotate Sample → Model Predictions → Metrics Logging (MLflow) → Alert if degraded
```
#### 2. Retraining & Redeployment

- (2.1) Automatic Retraining: If performance drops, the retraining pipeline is triggered using latest annotated data and historical data for stability [If needed we should prepare more dataset for model training]  
- (2.2) Evaluation: Retrained model is validated on all the available test sets. 
- (2.3) Deployment: Successful models are deployed automatically using MLOps Pipelines with blue-green deployment to avoid downtime.  

*Retraining & Deployment Flow*
```
Alert → MLflow Retraining Pipeline → Validation → Trail Deployment → Production
```