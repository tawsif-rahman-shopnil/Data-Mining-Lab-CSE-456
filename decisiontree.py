import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
df = pd.read_csv("weather.csv")

# Encode categorical columns to numerical values
label_encoders = {}
for column in ['Weather', 'Humidity', 'Wind', 'Play']:
    le = LabelEncoder()
    df[column + '_'] = le.fit_transform(df[column])
    label_encoders[column] = le

# Split the data into features (X) and target (y)
X = df[['Weather_', 'Humidity_', 'Wind_']]
y = df['Play']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=23)

# Create a Decision Tree classifier and train it on the training data
model = DecisionTreeClassifier(random_state=23)
model.fit(X_train, y_train)

# Evaluate the model on the test data
score = 100.0 * model.score(X_test, y_test)
print(f"Decision Tree prediction accuracy = {score:4.1f}%")

# Make predictions on the test data
result = model.predict(X_test)

# Print shapes of test data and predictions for debugging
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)
print("result shape:", result.shape)

# Display the classification report and confusion matrix
Labels = ['No', 'Yes']
print(classification_report(y_test, result, target_names=Labels))
print(confusion_matrix(y_test, result))

# Visualize the Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(model, feature_names=X.columns.tolist(), class_names=Labels, filled=True)
plt.show()
