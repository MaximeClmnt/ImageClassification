import sys,os

dirPath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dirPath)
import numpy as np
import matplotlib.pyplot as plt

from prepareData import lecture_cifar, decoupage_donnees, rgb2HOG, rgb2LBP
from kppv import kppv_distances, kppv_predict
from reseau import Reseau

def evaluation_classifieur(Ytest,Ypred):
    """
    renvoyantle taux de classification ()
    """
    
    Ncorrect = np.count_nonzero(Ytest.flatten()==Ypred.flatten())
    return Ncorrect/len(Ytest)



#%% Lecture et découpage données

X,Y = lecture_cifar(os.path.join(dirPath,"cifar-10-batches-py"), isTest = False)

### Utilisation de descripteurs ?

#X = rgb2HOG(X,cellSize=8,cellPerBlock=1)
#X = rgb2LBP(X)

### Découpage des données

Xapp,Yapp,Xtest,Ytest = decoupage_donnees(X, Y,taux=0.1)

#%% K plus proches voisins

#DistTest = kppv.kppv_distances(Xtest[:20], Xtest[:20])
#On verifie vien que la digonales de la matrice DistTest est nulle

#%%% Calcul de la matrice des distances (Xapp par rapport à Xtest)
Dist = kppv_distances(Xtest, Xapp)

k=7
Ypred = kppv_predict(Dist, Yapp, k)
accuracy = evaluation_classifieur(Ytest,Ypred)
print("Accuracy : " + str(accuracy))
#%%% Influence de k sur l'efficacité du classifieur 


liste_k = []
liste_accuracy = []
for k in range(1,101):
    Ypred = kppv_predict(Dist, Yapp, k)
    accu = evaluation_classifieur(Ytest,Ypred)
    liste_accuracy.append(accu)
    liste_k.append(k)

plt.plot(liste_k,liste_accuracy)

#%%% Validation croisée à N répertoires

N = 9

#Division de Xapp en N sous ensemble
foldSize = int(len(Xapp)/N)

liste_k = []
liste_accuracy = []
for n in range(N):
    foldXtest = Xapp[n*foldSize:(n+1)*foldSize]
    foldYtest = Yapp[n*foldSize:(n+1)*foldSize]
    foldXapp = np.concatenate((Xapp[:n*foldSize],Xapp[(n+1)*foldSize:]))
    foldYapp = np.concatenate((Yapp[:n*foldSize],Yapp[(n+1)*foldSize:]))
    
    foldDist = kppv_distances(foldXtest, foldXapp)
    
    liste_k.append([])
    liste_accuracy.append([])
    for k in range(1,101):
        foldYpred = kppv_predict(foldDist, foldYapp, k)
        accu = evaluation_classifieur(foldYtest,foldYpred)
        liste_accuracy[n].append(accu)
        liste_k[n].append(k)
        
plt.figure()
for n in range(N):
    plt.plot(liste_k[n][1:], liste_accuracy[n][1:])
plt.show()

accuracy_mean = np.mean(np.array(liste_accuracy),axis=0)

plt.figure()
plt.plot(liste_k[0][1:], accuracy_mean[1:])
plt.show()
#%% Reseau de neurone

np.random.seed(1) # pour que l'exécution soit déterministe
##########################
# Génération des données #
##########################
# N est le nombre de données d'entrée
# D_in est la dimension des données d'entrée
# D_h le nombre de neurones de la couche cachée
# D_out est la dimension de sortie (nombre de neurones de la couche de sortie)

# Création d'une matrice d'entrée Xnn et de sortie Ynn avec des valeurs aléatoires

loss_list = []
accu_list = []
accu_test_list = []

N = 45000
mini_batch_size = 45
N_mini_batch_per_epoch = int(N/mini_batch_size)
N_epoch = 100

Xnn = Xapp[:N]
Ynn = np.zeros((N,10))
Ynn[np.arange(N),Yapp[:N].flatten()]=1

N_test = 5000
Xnn_test = Xtest[:N_test]
Ynn_test = np.zeros((N_test,10))
Ynn_test[np.arange(N_test),Ytest[:N_test].flatten()]=1

Xnn_minibatches = [Xnn[k*mini_batch_size:(k+1)*mini_batch_size] for k in range(N_mini_batch_per_epoch)]
Ynn_minibatches = [Ynn[k*mini_batch_size:(k+1)*mini_batch_size] for k in range(N_mini_batch_per_epoch)]

D_in, D_out = Xnn.shape[1], 10
D_layers = [512,256]

learning_rate = 1e-4

reseau = Reseau(D_in, D_layers, D_out, activation='leaky_relu', taux_regul = 0.01)

for ne in range(10):
    for nmb in range(N_mini_batch_per_epoch):
        Xnn = Xnn_minibatches[nmb]
        Ynn = Ynn_minibatches[nmb]
        
        ####################################################
        # Passe avant : calcul de la sortie prédite Y_pred #
        ####################################################
        Y_pred = reseau.forward(Xnn)
        ############################################
        # Calcul et affichage de la fonction perte #
        ############################################
        
        #MSE loss
        #loss = np.square(Y_pred - Ynn).sum() / (2*mini_batch_size)
        #grad_Y_pred = (Y_pred - Ynn) / mini_batch_size
        
        #Cross entropy loss
        Y_pred_exp = np.exp(Y_pred)
        Y_pred = (Y_pred_exp.T/np.sum(Y_pred_exp,axis=1)).T
        loss = - np.sum( Ynn*np.log(Y_pred) ) / mini_batch_size
        
        grad_Y_pred = (Ynn * (Y_pred - 1) + (1-Ynn) * Y_pred) / mini_batch_size
        
        #regularissation
        if(True):
            loss += reseau.get_regularisation()
        
        loss_list.append(loss)
        accu=evaluation_classifieur(np.argmax(Ynn,axis=1),np.argmax(Y_pred,axis=1))
        accu_list.append(accu)
        if(nmb%10==0):
            print(str(ne)+"\t"+str(loss)+"\t"+str(accu))
        #####################################################
        # Passe arrière : calcul des gradients des Wi et bi #
        #####################################################
        reseau.backward(grad_Y_pred)
        ########################
        # Mise à jour de poids #
        ########################
        reseau.update_weights(learning_rate)
        
    
    Y_pred_test = reseau.forward(Xnn_test)
    accu_test=evaluation_classifieur(np.argmax(Ynn_test,axis=1),np.argmax(Y_pred_test,axis=1))
    accu_test_list.append(accu_test)
    if(ne%1==0):
        print(accu_test)
        