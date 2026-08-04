# MLOps
MLOps project, which predict the data using the Model

This project helps you learn to build and deploy an ML Model using a simple and real-world use case: predicting whether a person is diabetic based on health metrics. We’ll go from:

✅ Model Training
✅ Building the Model locally
✅ API Deployment with FastAPI
✅ Dockerization


Predict if a person is diabetic based on:

Pregnancies
Glucose 
Blood Pressure
BMI
Age

We use a Random Forest Classifier trained on the Pima Indians Diabetes Dataset.


1. Clone the Repo
git clone https://github.com/smsirajk/MLOps.git
cd first-mlops-project

3. Create Virtual Environment
python3 -m venv .mlops
source .mlops/bin/activate

5. Install Dependencies
pip install -r requirements.txt

Train the Model
python train.py

Run the API Locally
uvicorn main:app --reload


Sample Input for /predict
{
  "Pregnancies": 2,
  "Glucose": 130,
  "BloodPressure": 70,
  "BMI": 28.5,
  "Age": 45
}

Dockerize the API

Build the Docker Image

docker build -t diabetes-prediction-model .

Run the Container
docker run -p 8000:8000 diabetes-prediction-model
