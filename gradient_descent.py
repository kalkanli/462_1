from sklearn.preprocessing import OneHotEncoder
OHE=OneHotEncoder(sparse=False)
from scipy.special import softmax
import numpy as np

def calculateLoss(x,y,w):
        z = np.float128(x @ w)
        z=-z
        n=x.shape[0]
        loss=1/n * (np.trace(x @ w @ y.T) + np.sum(np.log(np.sum(np.exp(z),axis=1))))
        return loss

def calculateGradient(x,y,w):
        z=-x@w
        probs=softmax(z,axis=1)
        n=x.shape[0]
        gd=1/n * (x.T @ (y-probs)) #+ 2 * mu * W #regularization
        return gd 

def gradient_descent(X, Y, max_iter=10000, eta=0.001, loss_threshold=0.1):

	Y_onehot=OHE.fit_transform(Y.reshape(-1,1))
	W = np.zeros((X.shape[1], Y_onehot.shape[1]))
	step = 0
	while step < max_iter:
		step += 1
		gradient = calculateGradient(X, Y_onehot, W)
		W -= eta * gradient
		n_gradient=np.linalg.norm(gradient)
		print(n_gradient, step)
		if(n_gradient<loss_threshold):
			break
	return W

def predict (W ,H):
	Z = - H @ W 
	P = softmax(Z, axis=1)
	return np.argmax (P, axis=1) #reverse one hot encoding.