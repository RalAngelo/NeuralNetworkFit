import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

#Function to approach
def f(x):
    return np.cos(2*x) + x*np.sin(3*x) + x**0.5 - 2

a, b = 0, 5 #x element of [0, 5]
N = 100 #data size
X = np.linspace(a, b, N) #abscissa
Y = f(X)
X_train = X.reshape(-1,1)
Y_train = Y.reshape(-1,1)

#Network
modele = Sequential()
p = 10
modele.add(Dense(p, input_dim=1, activation='tanh'))
modele.add(Dense(p, activation='tanh'))
modele.add(Dense(p, activation='tanh'))
modele.add(Dense(p, activation='tanh'))
modele.add(Dense(1, activation='linear'))

#gradient descent method
mysgd = optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)
modele.compile(loss='mean_squared_error', optimizer=mysgd)
print(modele.summary())

history = modele.fit(X_train, Y_train, epochs=4000, batch_size=N)

Y_predict = modele.predict(X_train)
plt.plot(X_train, Y_train, color='blue')
plt.plot(X_train, Y_predict, color='red')
plt.show()

plt.plot(history.history['loss'])
plt.show()