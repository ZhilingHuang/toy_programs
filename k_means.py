from sklearn.cluster import KMeans
import numpy as np

def read_data():
    with open('/Users/hzl/Downloads/cape_town_destinations.csv') as f:
        lines = f.readlines()[1:10000]
        data = []
        for line in lines:
            if not line or line == '':
                continue
            lat, long, time = line.split(',')
            data.append([float(lat), float(long)])
    return data

data = read_data()
n_testing = 3000
traing_data = np.array(data[n_testing:])
testing_data = data[:n_testing]

n_clusters = 2000
model = KMeans(n_clusters=n_clusters, random_state=0).fit(data)

centroids = model.cluster_centers_

def distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

distance_loss = 0.0
centroid_index = model.predict(np.array(testing_data))

for index, centroid_index in enumerate(centroid_index):
    suboptimal_distance = distance(centroids[centroid_index], testing_data[index])
    if suboptimal_distance > 0.0008:
        print 'Trying points in cluster.'
        for index, d in enumerate(traing_data):
            if model.labels_[index] == centroid_index:
                suboptimal_distance = min(suboptimal_distance, distance(centroids[centroid_index], traing_data[index]))
    distance_loss += suboptimal_distance

average_loss = distance_loss / n_testing
print average_loss