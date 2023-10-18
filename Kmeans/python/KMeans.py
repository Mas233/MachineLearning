from copy import deepcopy
import numpy as np
from tqdm import tqdm


def _get_dist(vec1, vec2):
    v1=np.mat(vec1)
    v2=np.mat(vec2)
    return np.sqrt((v1-v2)*(v1-v2).T)


def _get_dist_square(vec1,vec2):
    v1=np.mat(vec1)
    v2=np.mat(vec2)
    return float((v1-v2)*(v1-v2).T)


def KMeans(data,clusters=3,iterations=20,max_delta=0.0001):
    samples,features=np.shape(data)
    print("Getting centers...")
    centers,dist=_get_center_and_dist_v2(data, clusters)

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
            center_delta+=_get_dist(centers[i], last_centers[i])
        # break condition: the centers do not change that much anymore
        if center_delta <= max_delta:break;

        # recalculate the distance
        progress_bar=tqdm(total=samples)
        for i in range(samples):
            progress_bar.update(1)
            progress_bar.set_description(f'Progress:{i}/{samples}')
            for j in range(clusters):
                dist[i][j]=_get_dist(data[i], centers[j])
        labels=np.argmin(dist,axis=1)
        progress_bar.close()
    return labels


# center generation Version 1: farthest possible point
def _get_center_and_dist(data, clusters):
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
            dist_to_centers[j][i-1]=_get_dist(data[j], centers[i - 1])
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
        dist_to_centers[j][clusters-1]=_get_dist(data[j], centers[clusters - 1])
    return centers,dist_to_centers


# centers generation Version 2: K-Means++ implementation
def _get_center_and_dist_v2(data, clusters):
    # using the kmeans++ implementation to get the centers
    samples,features=np.shape(data)
    centers=np.array(np.zeros((clusters,features)))
    center_index=[]
    dist_to_centers=np.array(np.zeros((samples,clusters)))
    probability_array=np.array(np.zeros(samples))

    # randomly choose the first center
    center_index.append(np.random.randint(0,samples))
    centers[0]=data[center_index[0]]

    # choose the rest centers
    for i in range(1,clusters):
        probability_sum=0
        for j in range(samples):
            dist_to_centers[j][i-1]=_get_dist(data[j], centers[i - 1])
            probability_array[j]=np.amin(dist_to_centers[j][:i])**2
            probability_sum+=probability_array[j]
        # make the probabilities add up to 1
        probability_array=probability_array/probability_sum

        # choose the next center point randomly
        center_index.append(np.random.choice(samples,p=probability_array))

        centers[i]=data[center_index[i]]
    for j in range(samples):
        dist_to_centers[j][clusters-1]=_get_dist(data[j], centers[clusters - 1])
    return centers,dist_to_centers