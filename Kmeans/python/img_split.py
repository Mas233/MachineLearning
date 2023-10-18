import numpy as np
import cv2
from KMeans import KMeans
# from sklearn.cluster import KMeans

# the count of sections the image should be seperated
cluster_cnt=3
# read the image
image = cv2.imread('crane.jpg')

# break the image into 2D array
pixels = image.reshape(-1, 3)

# construct KMeans object
kmeans = KMeans(n_clusters=cluster_cnt)
kmeans.fit(pixels)
labels = kmeans.labels_

# reconstructed segmented image
segmented_image = labels.reshape(image.shape[0], image.shape[1])

segments=[]

for i in range(cluster_cnt):
    segments.append(np.copy(image))
    segments[i][segmented_image != i] = 0

# display and save the segmented images
for i in range(cluster_cnt):
    cv2.imshow('Segment '+str(i), segments[i])
   # cv2.imwrite('Segment '+str(i)+'.png', segments[i])
cv2.waitKey(0)
cv2.destroyAllWindows()
