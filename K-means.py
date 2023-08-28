import pandas as pd
from sklearn_extra.cluster import KMedoids
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
df = pd.read_csv("clusters.csv")

# Extract the data columns A and B
data = df[['A', 'B']]

# Create a KMedoids instance with K=2
kmedoids = KMedoids(n_clusters=2, random_state=0)

# Fit the model to the data
kmedoids.fit(data)

# Get cluster assignments for each data point
labels = kmedoids.labels_

# Get cluster medoids (data points representing the cluster centers)
medoids_indices = kmedoids.medoid_indices_

# Print cluster medoids
print("Cluster Medoids:")
for medoid_index in medoids_indices:
    print(data.iloc[medoid_index])

# Visualize the clusters
plt.figure(figsize=(8, 6))
plt.scatter(data['A'], data['B'], c=labels, cmap='rainbow')
plt.scatter(data.iloc[medoids_indices]['A'], data.iloc[medoids_indices]['B'], marker='X', color='black', s=100, label='Medoids')
plt.xlabel('A')
plt.ylabel('B')
plt.title('K-Medoids Clustering')
plt.legend()
plt.show()
