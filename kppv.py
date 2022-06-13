import numpy as np

def kppv_distances(Xtest,Xapp,batchSize = 20):
    """

    Parameters
    ----------
    Xtest : Array(NtestxD)
        Matrice des données de test (Ntest nombre de données d'apprentissage', D dimension des données)
    
    Xapp : Array(NappxD)
        Matrice des données d'apprentiassage(N nombre de données de test, D dimension des données)

    Returns
    -------
    Dist : Array(Ntest,Napp)
        Matrice des distances l2 entre toutes les données de l'ensemble de test par rapport à toutes les données de l'ensemble d'apprentissage.

    """
    Dist = np.zeros((len(Xtest),len(Xapp)),dtype=('float32'))
    
    nBatchI = len(Xapp)//batchSize + 1
    nBatchJ = len(Xtest)//batchSize + 1
    for j in range(nBatchJ):
        batchStartJ = j*batchSize
        batchEndJ = (j+1)*batchSize
        #reshape de chaque batch pour pouvoir utiliser le broadcast de numpy
        testBatch = Xtest[batchStartJ:batchEndJ,None,:]
        
        print(str(j) + "/" + str(nBatchJ))
        for i in range(nBatchI):
            batchStartI = i*batchSize
            batchEndI = (i+1)*batchSize
            
            #reshape de chaque batch pour pouvoir utiliser le broadcast de numpy
            appBatch = Xapp[None,batchStartI:batchEndI,:]
            
            #squaredDiff est un tenseur où squaredDiff[j1,j2,i] = testBatch[j1,i] - appBatch[j2,i]
            squaredDiff = np.square(testBatch - appBatch) 
            Dist[batchStartJ:batchEndJ,batchStartI:batchEndI] = np.sqrt(np.sum(squaredDiff,axis=2))
    return Dist

def kppv_predict(Dist,Yapp,k):
    """
    Parameters
    ----------
    Dist : Array(Ntest,Napp)
        Matrice des distances l2 entre toutes les données de l'ensemble de test par rapport à toutes les données de l'ensemble d'apprentissage.
    Yapp : Array
        Vecteur des labels correspondant aux données d'apprentissage utilisées pour calculer Dist
    k : int
        Nombre de plus proche voisins a prendre encompte

    Returns
    -------
    Ypred : Array
        le vecteur des classes prédites pour les éléments de Xtest

    """
    minIndicies = np.argpartition(Dist,k,axis=1)[:,:k]
    #nearestDistances = np.take_along_axis(Dist,minIndicies,1)
    nearestLabels = np.reshape(Yapp[minIndicies],(len(Dist),k))
    
    # Calcul de label le plus present parmis les k plus proches voisins 
    # (Problème en cas d'égalité, possibilité de choisir les voisins les plus proches dans ce cas)
    # On reste avec une approche simple renvoyant le label d'indice le plus faible dans ce cas
    axis=1
    u, indices = np.unique(nearestLabels, return_inverse=True)
    Ypred = u[np.argmax(np.apply_along_axis(np.bincount, axis, indices.reshape(nearestLabels.shape),
                                    None, np.max(indices) + 1), axis=axis)]
    
    return Ypred
