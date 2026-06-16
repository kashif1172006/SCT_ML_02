import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('Mall_Customers.csv')
np.random.seed(42)
data = {
    'CustomerID': range(1, 201),
    'Recency': np.random.randint(0, 365, 200),      # Days since last purchase
    'Frequency': np.random.randint(1, 100, 200),    # Total purchases
    'Monetary': np.random.randint(100, 10000, 200)  # Total spend in $
}
df = pd.DataFrame(data)

X = df[['Recency', 'Frequency', 'Monetary']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

optimal_clusters = 3
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

cluster_analysis = df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
print("Cluster Profiles:")
print(cluster_analysis)

plt.figure(figsize=(10, 6))
plt.scatter(df['Frequency'], df['Monetary'], c=df['Cluster'], cmap='viridis', s=50)
plt.title('Customer Segments: Frequency vs Monetary Value')
plt.xlabel('Frequency')
plt.ylabel('Monetary Value ($)')
plt.colorbar(label='Cluster ID')
plt.show()