"""
This Python Model is related to Titanic Survival Prediction
This will predict whethere a passenger on the titanic survied or not 
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import os

dataset_path = "Titanic-Dataset.csv"
if not os.path.exists(dataset_path):
    print(f"File not found: {dataset_path}")
    exit()

titanic_df = pd.read_csv(dataset_path)

print("Initial Data Preview:")
print(titanic_df.head())
print("\nDataset Info:")
print(titanic_df.info())

columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
titanic_df = titanic_df.drop(columns=columns_to_drop, axis=1)

imputer = SimpleImputer(strategy='most_frequent')
titanic_df['Age'] = imputer.fit_transform(titanic_df[['Age']])
titanic_df['Embarked'] = imputer.fit_transform(titanic_df[['Embarked']]).ravel()

encoder = LabelEncoder()
titanic_df['Sex'] = encoder.fit_transform(titanic_df['Sex'])
titanic_df['Embarked'] = encoder.fit_transform(titanic_df['Embarked'])

print("\nProcessed Data Preview:")
print(titanic_df.head())

features = titanic_df.drop('Survived', axis=1)
target = titanic_df['Survived']

print("\nFeatures and Target Shapes:")
print("Features Shape:", features.shape)
print("Target Shape:", target.shape)

target = target.values.ravel()

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f"\nModel Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, predictions))

feature_importance_df = pd.DataFrame({'Feature': features.columns, 'Importance': classifier.feature_importances_})
print("\nFeature Importances:")
print(feature_importance_df.sort_values(by='Importance', ascending=False))
