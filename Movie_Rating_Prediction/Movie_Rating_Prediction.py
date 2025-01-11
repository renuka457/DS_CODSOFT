import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
# Load dataset
data = pd.read_csv('Dataset/IMDB_India_Movies.csv')

# Check the first few rows
print(data.head())

# Summary of dataset
print(data.describe())

# Check for missing values
print(data.isnull().sum())

# Fill missing values in 'Genre' and 'Director' with the most frequent value
data['Genre'].fillna(data['Genre'].mode()[0], inplace=True)
data['Director'].fillna(data['Director'].mode()[0], inplace=True)

# Fill missing ratings with the median
data['Rating'].fillna(data['Rating'].median(), inplace=True)

# Label encoding for categorical columns
label_encoder = LabelEncoder()
data['Genre'] = label_encoder.fit_transform(data['Genre'])
data['Director'] = label_encoder.fit_transform(data['Director'])

# Select features (X) and target (y)
X = data[['Genre', 'Director', 'Duration', 'Budget']]
y = data['Rating']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
