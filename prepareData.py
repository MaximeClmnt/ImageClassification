import numpy as np
import os

from skimage import io
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern
from skimage.feature import hog

def lecture_cifar(path,isTest=False):
    """
    Parameters
    ----------
    path : String
        Chemin vers le dossier des données cifar.
    isTest : Boolean, optional
        S'il s'agit d'un test on ne lit qu'un batch. The default is False.

    Returns
    -------
    X : Array(N,3072)
        Ensemble des vecteurs images (N = 10000 si isTest ,N = 50000 sinon)
    """
    nBatch = 5
    if isTest:
        nBatch = 1
        
    batchSize = 10000
        
    N = batchSize * nBatch
    D = 32 * 32 * 3
    
    X = np.zeros((N,D),dtype=('float32'))
    Y = np.zeros((N,1),dtype=('int'))
    
    import pickle
    for k in range(nBatch):
        with open(os.path.join(path,"data_batch_"+str(k+1)), 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            X[k*batchSize:(k+1)*batchSize,:] = dict[b'data']
            Y[k*batchSize:(k+1)*batchSize,0] = np.array(dict[b'labels'],dtype=('int'))
    X = X/256
    
    return X,Y

def decoupage_donnees(X,Y,taux=0.2):
    """
    Parameters
    ----------
    X : Array(NxD)
        Matrice des données (N nombre de données, D dimension des données)
        
    Y : Array(Nx1)
        Vecteur des labels (N nombre de données).
        
    taux : float, optional
        Part des données etant . The default is 0.2.

    Returns
    -------
    Xapp,Yapp : Array(Napp,D), Array(Napp,1) , Napp = N*(1-taux)
        Données et labeld d'apprentissage
    
    Xtest,Ytest : Array(Ntest,D), Array(Ntest,1) , Napp = N*taux
        Données et labeld de test

    """
    
    mask = np.ones(len(Y), dtype=bool)
    mask[:int(len(Y)*taux)] = False
    np.random.shuffle(mask)
    
    Xapp = X[mask,:]
    Yapp = Y[mask]
    Xtest = X[mask==False,:]
    Ytest = Y[mask==False]
    return Xapp,Yapp,Xtest,Ytest

def rgb2LBP(X,n_points = 16, radius = 2):
    """
    Parameters
    ----------
    X : Array(NxD)
        Matrice des données à convertir (N nombre de données, D dimension des données)
    n_points : int, optional
        Nombre de point à prendre en compte pour le calcul des motifs binaires locaux. The default is 16.
    radius : float, optional
        Rayon du cercle à prendre en compte pour le calcul. The default is 2.

    Returns
    -------
    newX : Array(NxD2)
        Matrice des données converties (N nombre de données, D2 dimension des données converties)

    """
    reshaped = np.reshape(X, (len(X),32,32,3),'F')
    newX = np.zeros((len(X),18))
    for i,x in enumerate(reshaped):
        img = rgb2gray(x)
        lbp = local_binary_pattern(img, n_points, radius, 'uniform')
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
        newX[i,:] = hist
    return newX

def rgb2HOG(X,orientations=8,cellSize=4,cellPerBlock=1):
    """
    Parameters
    ----------
    X : Array(NxD)
        Matrice des données à convertir (N nombre de données, D dimension des données)
    orientations : int, optional
        Nombre d'orientations du gradient. The default is 8.
    cellSize : int, optional
        Taille des cellules dans lesquels on calcul le gradient. The default is 4.
    cellPerBlock : int, optional
        Nombre de cellules par blocs. The default is 1.
    Returns
    -------
    newX : Array(NxD2)
        Matrice des données converties (N nombre de données, D2 dimension des données converties)

    """
    reshaped = np.reshape(X, (len(X),32,32,3),'F')
    
    hogTest = hog(reshaped[0], orientations=orientations, pixels_per_cell=(cellSize, cellSize), cells_per_block=(cellPerBlock, cellPerBlock), block_norm='L2-Hys', feature_vector=True, multichannel=True)
    newX = np.zeros((len(X),len(hogTest)))
    
    for i,x in enumerate(reshaped):
        hogImg = hog(x, orientations=orientations, pixels_per_cell=(cellSize, cellSize), cells_per_block=(cellPerBlock, cellPerBlock), block_norm='L2-Hys', feature_vector=True, multichannel=True)
        newX[i,:] = hogImg
    return newX

