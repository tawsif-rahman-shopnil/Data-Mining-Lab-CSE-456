import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

# Read the CSV file into a DataFrame
df = pd.read_csv("cars.csv")

# Encode categorical columns to numerical values
label_encoders = {}
for column in ['Color', 'Type', 'Origin', 'Stolen']:
    le = LabelEncoder()
    df[column + '_cat'] = le.fit_transform(df[column])
    label_encoders[column] = le

# Split the data into features (X) and target (y)
X = df[['Color_cat', 'Type_cat', 'Origin_cat']]
y = df['Stolen_cat']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Naive Bayes classifier and train it on the training data
model = GaussianNB()
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Create a confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix as a heatmap
sns.set(font_scale=1.2)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoders['Stolen'].classes_, yticklabels=label_encoders['Stolen'].classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Make predictions on a new car scenario: {Red, Domestic, Suv}
new_car = [[label_encoders['Color'].transform(['Red'])[0],
            label_encoders['Type'].transform(['Suv'])[0],
            label_encoders['Origin'].transform(['Domestic'])[0]]]

predicted_outcome = model.predict(new_car)

if predicted_outcome[0] == label_encoders['Stolen'].transform(['Yes'])[0]:
    outcome = 'Stolen'
else:
    outcome = 'Not Stolen'

print(f"Predicted outcome for the car: {outcome}")
