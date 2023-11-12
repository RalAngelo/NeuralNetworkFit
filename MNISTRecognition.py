import numpy as np
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Telechargement des donnees
(X_train_data,Y_train_data),(X_test_data,Y_test_data) = mnist.load_data()

N = X_train_data.shape[0] # N = 60 000 donnees
# Donnees d'apprentissage X
X_train = np.reshape(X_train_data,(N,784)) # vecteur image
X_train = X_train/255 # normalisation

# Donnees d'apprentissage Y vers une liste de taille 10
Y_train = to_categorical(Y_train_data, num_classes=10)

# Donnees de test
X_test = np.reshape(X_test_data,(X_test_data.shape[0],784))
X_test = X_test/255
Y_test = to_categorical(Y_test_data, num_classes=10)


### Le reseau de neurones
p = 8
modele = Sequential()

# Premiere couche : p neurones (entree de dimension 784 = 28x28)
modele.add(Dense(p, input_dim=784, activation='sigmoid'))

# Deuxieme couche : p neurones
modele.add(Dense(p, activation='sigmoid'))

# Couche de sortie : 1O neurones (un par chiffre)
modele.add(Dense(10, activation='softmax'))

# Choix de la methode de descente de gradient
modele.compile(loss='categorical_crossentropy',
optimizer='sgd',
metrics=['accuracy'])
print(modele.summary())


### Calcul des poids par descente de gradient
modele.fit(X_train, Y_train, batch_size=32, epochs=40)


### Resultats
resultat = modele.evaluate(X_test, Y_test, verbose=0)
print('Valeur de l''erreur sur les donnees de test (loss):', resultat[0])
print('Precision sur les donnees de test (accuracy):', resultat[1])
