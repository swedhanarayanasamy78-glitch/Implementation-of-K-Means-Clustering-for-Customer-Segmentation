# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Initialize K cluster centroids randomly from the dataset.

2.Assign each customer to the nearest centroid based on distance (e.g., Euclidean).

3.Recalculate centroids as the mean of all assigned points in each cluster.

4.Repeat assignment and centroid update until centroids do not change.

5.Output final clusters representing customer segments.
## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: SWEDHA.N
RegisterNumber: 212225230280 
*/
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

data = pd.read_csv(r"C:\Users\acer\Downloads\CustomerData.csv")

print(data.head())
print(data.columns)

features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = data[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertia_values = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)  # Explicit n_init to suppress warning
    kmeans.fit(X_scaled)
    inertia_values.append(kmeans.inertia_)
    
plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), inertia_values, marker='o', linestyle='-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.show()

optimal_clusters = 4
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)  # Explicit n_init
kmeans.fit(X_scaled)

data['Cluster'] = kmeans.labels_

sil_score = silhouette_score(X_scaled, kmeans.labels_)
print(f'Silhouette Score: {sil_score}')

print("\nName: B MARLIYA BANU")
print("Reg No.: 212225040229\n")
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data,x='Annual Income (k$)',y='Spending Score (1-100)',hue='Cluster', palette='viridis',s=100,alpha=0.7)
plt.title('Customer Segmentation based on Annual Income and Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.show()

```

## Output:
<img width="732" height="563" alt="image" src="https://github.com/user-attachments/assets/1073f07b-fdf3-4e8e-891a-3d84a0279ed9" />

<img width="837" height="512" alt="image" src="https://github.com/user-attachments/assets/13c06895-ae4c-4f87-a98b-602cb97f1553" />

## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
