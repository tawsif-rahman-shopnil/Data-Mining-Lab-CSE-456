import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
df = pd.read_csv("foodtype.csv")

# Encode categorical columns to numerical values
label_encoders = {}
for column in ['Ingredient', 'Food type']:
    le = LabelEncoder()
    df[column + '_cat'] = le.fit_transform(df[column])
    label_encoders[column] = le

# Split the data into features (X) and target (y)
X = df[['Sweet', 'Crunch']].values
y = df['Food type_cat']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a KNeighborsClassifier instance with K=3
knn = KNeighborsClassifier(n_neighbors=3)

# Fit the model to the training data
knn.fit(X_train, y_train)

# Predict the food type for a new ingredient (Sweet=6, Crunch=4)
new_ingredient = [[6, 4]]
predicted_food_type = knn.predict(new_ingredient)

predicted_food_type_label = label_encoders['Food type'].inverse_transform(predicted_food_type)[0]
print(f"Predicted Food Type for the ingredient: {predicted_food_type_label}")

# Visualize the clusters
plt.figure(figsize=(8, 6))
for food_type in df['Food type_cat'].unique():
    plt.scatter(df[df['Food type_cat'] == food_type]['Sweet'], df[df['Food type_cat'] == food_type]['Crunch'], label=label_encoders['Food type'].inverse_transform([food_type])[0])
plt.scatter(new_ingredient[0][0], new_ingredient[0][1], marker='x', color='red', label='New Ingredient')
plt.xlabel('Sweet')
plt.ylabel('Crunch')
plt.title('Food Type Clusters')
plt.legend()
plt.show()
