import numpy as np

class Reseau:
    
    def __init__(self, D_in, D_layers, D_out, activation='sigmoid', taux_regul = 0):
        """
        Parameters
        ----------
        D_in : int
            Dimension des données d'entrée'
        D_layers : int[]
            Nombre de neurone des couches cachées.
        D_out : int
            Dimension de sortie (nombre de neurones de la couche de sortie)
        activation : String ('sigmoid', 'relu'), optional
            Fonction d'activation à utiliser. The default is 'sigmoid'.

        Returns
        -------
        None.

        """
        self.m_D_in = D_in
        self.m_D_layers = D_layers
        self.m_D_Dout = D_out
        
        #Initialisation des couches
        self.m_layers = []
        self.m_layers.append(Layer(D_in, D_layers[0], activation, taux_regul))
        for k in range(len(D_layers)-1):
            self.m_layers.append(Layer(D_layers[k], D_layers[k+1], activation, taux_regul))
        self.m_layers.append(Layer(D_layers[-1], D_out, activation, taux_regul))
        
        
    def forward(self, X):
        """
        Calcul Ypred en fonction de X en appelant successivement la fonction forward correspondant à la fonction d'activation souhaitée de chaque couche du reseau.
        Chaque couche enregistre le resultat intermédiaire pour le calcul du gradient.

        Parameters
        ----------
        X : Array(N,D_in)
            Matrice d'entrée correspondant à l'enssemble des données à classer

        Returns
        -------
        Ypred : Array(N,D_out)
            Matrice de prediction des labels pour chaque entrée

        """
        Xi = X
        for l in self.m_layers:
            Oi = l.m_forward(l,Xi)
            
            #L'entrée de la couche suivante est la sortie de la couche actuelle
            Xi = Oi.copy()
        Ypred = Oi
        return Ypred
    
    def backward(self, grad_Y_predf):
        """
        Calcul l'ensemeble des gradient utiles en fonction de gras_Y_predf et des données caluclées lors de l'appel à forward.
        Chaque couche enregistre les resultats correspondants au gradients des poids nécéssaire pour la mise à jour de ces derniers
        Parameters
        ----------
        grad_Y_predf : Array(N,D_in)
            Gradient de Y_pred par rapport à la loss

        Returns
        -------
        None

        """
        grad_Oif = grad_Y_predf
        for l in reversed(self.m_layers):
            grad_Xif = l.m_backward(l,grad_Oif)
            #le gradient de 
            grad_Oif = grad_Xif.copy()
    
    def update_weights(self,learning_rate):
        """
        Mets à jours les poids en fonction des gradients calculés à l'appel de backward.
        
        Parameters
        ----------
        learning_rate : 
            taux d'apprentissage'           

        Returns
        -------
        None

        """
        for l in self.m_layers:
            l.update_weights(learning_rate)
    
    def get_regularisation(self):
        """
        Returns
        -------
        float :
            Somme de l'ensemble du carré des poids du reseau.

        """
        regul = 0
        for l in self.m_layers:
            regul += l.get_regularisation()
        return regul


class Layer:
    
    def __init__(self, D_Xi, D_Oi , activation='sigmoid', taux_regul = 0):
        """
        Parameters
        ----------
        D_Xi : int
            Dimension de l'entrée de la couche.
        D_Oi : int
            Dimension de l'a sortie de la couche.
        activation : String ('sigmoid', 'relu')
            Fonction d'activation à utiliser (default = 'sigmoid'')

        Returns
        -------
        None.

        """
        self.m_D_Xi = D_Xi
        self.m_D_Oi = D_Oi
        self.m_activation = activation
        self.m_taux_regul = taux_regul
        
        if(activation == 'relu'):
            self.m_forward = Layer.relu_forward
            self.m_backward = Layer.relu_backward
            self.relu_init(D_Xi,D_Oi)
        elif(activation == 'leaky_relu'):
            self.m_forward = Layer.leaky_relu_forward
            self.m_backward = Layer.leaky_relu_backward
            self.relu_init(D_Xi,D_Oi)
        else: ## activation = 'sigmoid' ?
            self.m_forward = Layer.sigmoid_forward
            self.m_backward = Layer.sigmoid_backward
            self.sigmoid_init(D_Xi,D_Oi)
            
        
    def update_weights(self,learning_rate):
        """
        Modifie les poids de la couche en fonction des gradients caluclés et du taux d'apprentissage'

        Parameters
        ----------
        learning_rate : float
            Taux d'apprentissage'

        Returns
        -------
        None
        """
        self.m_Wi = self.m_Wi - self.m_grad_Wif * learning_rate
        self.m_bi = self.m_bi - self.m_grad_bif * learning_rate
        return 0  
        
    
    def sigmoid_forward(self, Xi):
        """
        Applique la fonction sigmoïd sur le potentiel d'entrée de la couche cachée Ii
        
        Parameters
        ----------
        Xi : Array
            Matrice d'entrée de la couche'

        Returns
        -------
        Array
            Matrice de sortie Oi pour la fonction d'activation "sigmoïd"
        """
        #Commun
        self.m_Xi = Xi
        self.m_Ii = Xi.dot(self.m_Wi) + self.m_bi
        
        # Activation
        self.m_Oi = 1/(1+np.exp(-self.m_Ii))
        return self.m_Oi
    
    def sigmoid_backward(self, grad_Oif):
        """
        Calcul les gradients des poids (Wi et bi) et de l'entrée (Xi) à partir du gradient de la sortie(Oi) pour la fonction d'activation "sigmoid".
        Le gradient des poids est modifié pour prendre en compte la régularisation si besoin.
        
        Parameters
        ----------
        grad_Oif : Array
            Matrice du gradient de la sortie Oi par rapport à la loss.

        Returns
        -------
        Array
            Matrice du gradient de la sortie Oi par rapport à la loss pour la fonction d'activation "sigmoid".
        """
        # Activation
        self.m_grad_Iif = (1-self.m_Oi)*self.m_Oi * grad_Oif
        
        #Commun
        self.m_grad_bif = np.ones((len(self.m_Xi),1)).T.dot(self.m_grad_Iif)
        self.m_grad_Wif = self.m_Xi.T.dot(self.m_grad_Iif)
        #Regularisation
        if(self.m_taux_regul):
            self.m_grad_bif += self.m_bi * self.m_taux_regul
            self.m_grad_Wif += self.m_Wi * self.m_taux_regul
        
        self.m_grad_Xif = self.m_grad_Iif.dot(self.m_Wi.T)
        return self.m_grad_Xif
    
    def sigmoid_init(self,D_Xi,D_Oi):
        """
        Initialisation des poids ajustée pour la fonction d'activation "sigmoïd"'

        Parameters
        ----------
        D_Xi : int
            Dimension du vecteur d'entrée de la couche.
        D_Oi : int
            Dimension du vecteur de sortie de la couche.

        Returns
        -------
        None.
        """
        # W est la matrice des poids de chaque sinaps de la couche
        # b est la matrice des bias en entrée
        self.m_Wi = np.random.randn(D_Xi, D_Oi) / np.sqrt(D_Xi)
        self.m_bi = np.zeros((1,D_Oi))
        
    
    def relu_forward(self, Xi):
        """
        Applique la fonction relu sur le potentiel d'entrée de la couche cachée Ii
        
        Parameters
        ----------
        Xi : Array
            Matrice d'entrée de la couche'

        Returns
        -------
        Array
            Matrice de sortie Oi pour la fonction d'activation "relu""
        """
        #Commun
        self.m_Xi = Xi
        self.m_Ii = Xi.dot(self.m_Wi) + self.m_bi
        
        # Activation
        self.m_Oi = np.maximum(self.m_Ii,0)
        return self.m_Oi
    
    def relu_backward(self, grad_Oif):
        """
        Calcul les gradients des poids (Wi et bi) et de l'entrée (Xi) à partir du gradient de la sortie(Oi) pour la fonction d'activation "relu".
        Le gradient des poids est modifié pour prendre en compte la régularisation si besoin.
        
        Parameters
        ----------
        grad_Oif : Array
            Matrice du gradient de la sortie Oi par rapport à la loss.

        Returns
        -------
        Array
            Matrice du gradient de la sortie Oi par rapport à la loss pour la fonction d'activation "relu".
        """
        #Activation
        self.m_grad_Iif = (self.m_Oi>0) * grad_Oif
        
        #Commun
        self.m_grad_bif = np.ones((len(self.m_Xi),1)).T.dot(self.m_grad_Iif)
        self.m_grad_Wif = self.m_Xi.T.dot(self.m_grad_Iif)
        #Regularisation
        if(self.m_taux_regul):
            self.m_grad_bif += self.m_bi * self.m_taux_regul
            self.m_grad_Wif += self.m_Wi * self.m_taux_regul
        
        self.m_grad_Xif = self.m_grad_Iif.dot(self.m_Wi.T)
        return self.m_grad_Xif
    
    def relu_init(self,D_Xi,D_Oi):
        """
        Initialisation des poids ajustée pour la fonction d'activation "relu" et "leaky-relu"'

        Parameters
        ----------
        D_Xi : int
            Dimension du vecteur d'entrée de la couche.
        D_Oi : int
            Dimension du vecteur de sortie de la couche.

        Returns
        -------
        None.
        """
        # W est la matrice des poids de chaque sinaps de la couche
        # b est la matrice des bias en entrée
        self.m_Wi = np.random.randn(D_Xi, D_Oi) * np.sqrt(2/D_Xi)
        self.m_bi = np.zeros((1,D_Oi)) + 0.001
        
    def leaky_relu_forward(self, Xi):
        """
        Applique la fonction leaky relu sur le potentiel d'entrée de la couche cachée Ii
        
        Parameters
        ----------
        Xi : Array
            Matrice d'entrée de la couche'

        Returns
        -------
        Array
            Matrice de sortie Oi pour la fonction d'activation "leaky-relu""
        """
        #Commun
        self.m_Xi = Xi
        self.m_Ii = Xi.dot(self.m_Wi) + self.m_bi
        
        # Activation
        self.m_Oi = np.maximum(self.m_Ii,0) + np.minimum(self.m_Ii*0.01,0)
        return self.m_Oi
    
    def leaky_relu_backward(self, grad_Oif):
        """
        Calcul les gradients des poids (Wi et bi) et de l'entrée (Xi) à partir du gradient de la sortie(Oi) pour la fonction d'activation "leaky-relu".
        Le gradient des poids est modifié pour prendre en compte la régularisation si besoin.
        
        Parameters
        ----------
        grad_Oif : Array
            Matrice du gradient de la sortie Oi par rapport à la loss.

        Returns
        -------
        Array
            Matrice du gradient de la sortie Oi par rapport à la loss pour la fonction d'activation "leaky-relu".
        """
        #Activation
        self.m_grad_Iif = ((self.m_Oi>0) + (self.m_Oi<0)*0.01 ) * grad_Oif 
        
        #Commun
        self.m_grad_bif = np.ones((len(self.m_Xi),1)).T.dot(self.m_grad_Iif)
        self.m_grad_Wif = self.m_Xi.T.dot(self.m_grad_Iif)
        #Regularisation
        if(self.m_taux_regul):
            self.m_grad_bif += self.m_bi * self.m_taux_regul
            self.m_grad_Wif += self.m_Wi * self.m_taux_regul
        
        self.m_grad_Xif = self.m_grad_Iif.dot(self.m_Wi.T)
        return self.m_grad_Xif
    
    def get_regularisation(self):
        """
        Returns
        -------
        float
            Somme des poids au carré de la couche.

        """
        if(self.m_taux_regul):
            return np.square(self.m_Wi).sum()*self.m_taux_regul/2 + np.square(self.m_bi).sum()*self.m_taux_regul/2
        else:
            return 0
