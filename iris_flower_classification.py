import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

iris_df = pd.read_csv('IRIS.csv')

print(iris_df.head())
X = iris_df.drop('species', axis=1)
y = iris_df['species']

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(random_state=42, n_estimators=100)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

joblib.dump(clf, 'iris_classifier_model.pkl')

model = joblib.load('iris_classifier_model.pkl')
sample_data = [[5.1, 3.5, 1.4, 0.2]]
sample_prediction = model.predict(sample_data)
predicted_species = label_encoder.inverse_transform(sample_prediction)
print("\nPredicted Species:", predicted_species[0])
