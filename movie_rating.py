"""
This is the Python Model which will help in predicting the
rating of the movie based on features like director, ratings ..etc
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

movie_df = pd.read_csv("IMDb Movies India.csv", encoding='ISO-8859-1')

print("Initial Data Preview:")
print(movie_df.head())

movie_df['Year'] = movie_df['Year'].str.extract(r'(\d{4})').astype(float)
movie_df['Duration'] = movie_df['Duration'].str.extract(r'(\d+)').astype(float)

movie_df['Votes'] = pd.to_numeric(movie_df['Votes'], errors='coerce')

imputer = SimpleImputer(strategy='most_frequent')
movie_df['Rating'] = imputer.fit_transform(movie_df[['Rating']])
movie_df['Votes'] = imputer.fit_transform(movie_df[['Votes']])

categorical_features = ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

features = movie_df.drop(['Name', 'Rating'], axis=1)
target = movie_df['Rating']

features_transformed = preprocessor.fit_transform(features)

feature_imputer = SimpleImputer(strategy='most_frequent')
features_transformed = feature_imputer.fit_transform(features_transformed)

X_train, X_test, y_train, y_test = train_test_split(features_transformed, target, test_size=0.2, random_state=42)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

predictions = regressor.predict(X_test)


mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Evaluation:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")


plt.figure(figsize=(8, 5))
sns.histplot(movie_df['Rating'], kde=True, bins=10)
plt.title("Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.savefig("rating_distribution.png") 
plt.close()

plt.figure(figsize=(8, 5))
sns.scatterplot(x=movie_df['Votes'], y=movie_df['Rating'])
plt.title("Votes vs Rating")
plt.xlabel("Votes")
plt.ylabel("Rating")
plt.savefig("votes_vs_rating.png") 
plt.close()
