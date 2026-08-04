import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib


# Load the dataset from working source 

url_git = "https://github.com/plotly/datasets/blob/master/diabetes.csv"
df = pd.read_csv(url_git)

print("Dataset loaded successfully.")
print(f"Columns :",df.columns.tolist())

#Prepare the data for training
x = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split the dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)   

# train the model 

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#Save the trained model to a file
joblib.dump(model, 'diabetes_rf_model.pkl')

print("Model trained and saved successfully as 'diabetes_rf_model.pkl'.")


