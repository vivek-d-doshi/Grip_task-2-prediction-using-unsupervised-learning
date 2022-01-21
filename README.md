<img align = right height = 120 width = 120 src = https://www.thesparksfoundationsingapore.org/images/logo_small.png>

# Grip_task-2-prediction-using-unsupervised-learning


This repository contains the tasks that I completed while working as an intern for [The Sparks Foundation.](https://www.thesparksfoundationsingapore.org/)
- **Internship Category** - Data Science and Business Analytics
- **Internship Duration** - 1 Month ( October-2020 )
- **Internship Type** - Work from Home

## Code

```bash
#importing libraries

import pandas as pd
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
```

```bash
#Loading and create the dataFrame from data set
df = pd.read_csv("iris.csv")
df.shape

# Finding the optimum number of clusters for k-means classification
x = df.iloc[:, [1,2,3,4]].values

```
```bash
#visulizing the clusters using line graph
from sklearn.cluster import KMeans
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++',max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
# Plotting the results onto a line and scatter graph
plt.plot(range(1, 11), wcss)
plt.scatter(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') # Within cluster sum of squares
plt.show()

```
![28c434b3-5816-4246-8085-235961327f2f](https://user-images.githubusercontent.com/77320499/150575729-bd0d89b5-fbe2-489c-a1d8-fe444966de39.png)
```bash

# Applying kmeans to the dataset / Creating the kmeans classifier
kmeans = KMeans(n_clusters = 3, init = 'k-means++',
                max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)

plt.figure(figsize=(10,8))

# Visualising the clusters - On the first two columns
plt.subplot(2, 2, 1)
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = 'violet', label = 'Iris-setosa')
plt.legend()

plt.subplot(2, 2, 2)
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'purple', label = 'Iris-versicolour')
plt.legend()

plt.subplot(2, 2, 3)
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = 'skyblue', label = 'Iris-virginica')
plt.legend()

# Plotting the centroids of the clusters
plt.subplot(2, 2, 4)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'yellow', label = 'Centroids')
plt.legend()

```
![001837c5-3145-43ef-9371-e4ee0e8df8d6](https://user-images.githubusercontent.com/77320499/150575872-b7dfbce3-1e1a-4eb2-8bb6-1c6be64fa9aa.png)
```bash
plt.figure(figsize=(10,8))

# Visualising the clusters - On the first two columns
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = 'violet', label = 'Iris-setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'purple', label = 'Iris-versicolour')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = 'skyblue', label = 'Iris-virginica')

# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'yellow', label = 'Centroids')
plt.legend()

```
![7df16b94-9f75-4eaa-bf82-0f99c7ab4443](https://user-images.githubusercontent.com/77320499/150575920-d207ab6b-80ce-4aff-9b8e-c750c23e82df.png)
