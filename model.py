import numpy as np

def distance(v1,v2):
    return np.sqrt(((v1-v2)**2).sum())

def knn(train,test,k=5):
    dist = []
    
    for i in range(train.shape[0]):
        train_x = train[i,:-1]
        train_y = train[i,-1]
        d = distance(test,train_x)
        dist.append([d,train_y])
    dk = sorted(dist, key= lambda x: x[0])[:k]
    labels = np.array(dk)[:,-1]
    output = np.unique(labels, return_counts= True)
    index = np.argmax(output[1])
    return output[0][index]