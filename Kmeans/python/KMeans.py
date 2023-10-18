from copy import deepcopy
import numpy as np
from tqdm import tqdm

def _getDist(vec1,vec2):
    v1=np.mat(vec1)
    v2=np.mat(vec2)
    return np.sqrt((v1-v2)*(v1-v2).T)

def KMeans(data,clusters=3,iterations=300,max_delta=0.0001):
    samples,features=np.shape(data)
    print("Getting centers...")
    centers,dist=_getCenterIndex_Center_Dist(data,clusters)

    # start clustering for iterations times or until the clustering centers do not change anymore
    print("Start clustering...")
    epoch=0
    center_delta=0
    labels=np.array(np.zeros(samples))

    # first clustering, the distance is already calculated
    labels=np.argmin(dist,axis=1)
    # start clustering
    while epoch<iterations:
        epoch+=1
        center_delta=0
        # recalculate the centers
        last_centers=deepcopy(centers)
        print(f'running the {epoch}th iteration')
        for i in range(clusters):
            centers[i]=np.mean(data[labels==i],axis=0)
            center_delta+=_getDist(centers[i],last_centers[i])
        # break condition: the centers do not change that much anymore
        if center_delta <= max_delta:break;

        # recalculate the distance
        progress_bar=tqdm(total=samples)
        for i in range(samples):
            progress_bar.update(1)
            progress_bar.set_description(f'Progress:{i}/{samples}')
            for j in range(clusters):
                dist[i][j]=_getDist(data[i],centers[j])
        labels=np.argmin(dist,axis=1)
        progress_bar.close()
    return labels


def _getCenterIndex_Center_Dist(data,clusters):
    # using the kmeans++ implementation to get the centers
    samples,features=np.shape(data)
    centers=np.array(np.zeros((clusters,features)))
    center_index=[]
    dist_to_centers=np.array(np.zeros((samples,clusters)))

    # randomly choose the first center
    center_index.append(np.random.randint(0,samples))
    centers[0]=data[center_index[0]]

    # choose the rest centers
    for i in range(1,clusters):
        for j in range(samples):
            dist_to_centers[j][i-1]=_getDist(data[j],centers[i-1])
        # exclude center points to avoid duplicate
        mask = np.ones(dist_to_centers.shape[0], dtype=bool)
        mask[center_index] = False
        rest_arr=dist_to_centers[mask]
        # get the index of the farthest point from the rest points
        center_index.append(
            np.where(mask)[0][
                np.argmax(np.sum(rest_arr,axis=1))
            ])
        centers[i]=data[center_index[i]]
    for j in range(samples):
        dist_to_centers[j][clusters-1]=_getDist(data[j],centers[clusters-1])
    return centers,dist_to_centers
