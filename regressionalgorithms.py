from __future__ import division  # floating point division
import numpy as np
import math
import time
import utilities as utils

class Regressor:
    """
    Generic regression interface; returns random regressor
    Random regressor randomly selects w from a Gaussian distribution
    """

    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}
        self.reset(parameters)

    def reset(self, parameters):
        """ Reset learner """
        self.weights = None
        self.resetparams(parameters)

    def resetparams(self, parameters):
        """ Can pass parameters to reset with new parameters """
        self.weights = None
        try:
            utils.update_dictionary_items(self.params,parameters)
        except AttributeError:
            # Variable self.params does not exist, so not updated
            # Create an empty set of params for future reference
            self.params = {}

    def getparams(self):
        return self.params

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.weights = np.random.rand(Xtrain.shape[1])

    def predict(self, Xtest):
        """ Most regressors return a dot product for the prediction """
        ytest = np.dot(Xtest, self.weights)
        return ytest

class RangePredictor(Regressor):
    """
    Random predictor randomly selects value between max and min in training set.
    """

    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.min = 0
        self.max = 1
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.min = np.amin(ytrain)
        self.max = np.amax(ytrain)

    def predict(self, Xtest):
        ytest = np.random.rand(Xtest.shape[0])*(self.max-self.min) + self.min
        return ytest

class MeanPredictor(Regressor):
    """
    Returns the average target value observed; a reasonable baseline
    """
    def __init__( self, parameters={} ):
        self.params = {}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.mean = None
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.mean = np.mean(ytrain)

    def predict(self, Xtest):
        return np.ones((Xtest.shape[0],))*self.mean


class FSLinearRegression(Regressor):
    """
    Linear Regression with feature selection, and ridge regularization
    """
    def __init__( self, parameters={} ):
        self.params = {'features': [1,2,3,4,5]}
        self.reset(parameters)

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples
        numsamples = Xtrain.shape[0]
        Xless = Xtrain[:,self.params['features']]
        self.weights = np.dot(np.dot(np.linalg.pinv(np.dot(Xless.T,Xless)/numsamples), Xless.T),ytrain)/numsamples

    def predict(self, Xtest):
        Xless = Xtest[:,self.params['features']]
        ytest = np.dot(Xless, self.weights)
        return ytest

class RidgeLinearRegression(Regressor):
    """
    Linear Regression with ridge regularization (l2 regularization)
    TODO: currently not implemented, you must implement this method
    Stub is here to make this more clear
    Below you will also need to implement other classes for the other algorithms
    """
    def __init__( self, parameters={}):
        # Default parameters, any of which can be overwritten by values passed to params
        self.weights = None
        #self.lamb =  self.parameters['lamb']
        self.params = {'features': [1, 2, 3, 4, 5],'lamb':0.01}
        self.reset(parameters)
        self.lamb = self.params['lamb']

    def learn(self, Xtrain, ytrain):
        numsamples = Xtrain.shape[0]
        Xless = Xtrain[:, self.params['features']]
        sizeI = Xless.shape[1]
        self.weights = np.dot(np.dot(np.linalg.inv(np.dot(Xless.T, Xless)  + np.dot(self.lamb,np.eye(sizeI)) / numsamples), Xless.T), ytrain)

    def predict(self, Xtest):
        Xless = Xtest[:,self.params['features']]
        ytest = np.dot(Xless,self.weights)
        return ytest

class LassoLinearRegression(Regressor):
    def __init__(self,parameters={}):
        self.weights = 0
        self.params = {'features':[1,2,3]}
        self.reset(parameters)

    def reset(self, parameters):
        self.params = parameters

    def learn(self, Xtrain, ytrain):
        error = 10000
        tolerance = 10e-4
        lamb = self.params['regwgt']

        Xless = Xtrain[:, self.params['features']]
        numsamples = Xtrain.shape[0]

        #precomputing
        XX = np.dot(np.dot(Xless.T,Xless),1/numsamples)
        Xy = np.dot(np.dot(Xless.T,ytrain),1/numsamples)

        #stepsize
        stepsize = 1/(2*np.linalg.norm(XX))

        #init c(w)
        self.weights = np.zeros(Xless.shape[1])
        cw = np.square(np.linalg.norm(np.dot(Xless,self.weights)-ytrain)) + lamb * np.linalg.norm(self.weights,ord = 1)

        #define max iteration
        iter = 0
        maxIter = 5000
        while np.abs(np.subtract(cw , error)) > tolerance and (iter < maxIter):
            error = cw
            self.weights = self.proximalOperator( (self.weights - stepsize*np.dot(XX,self.weights) + stepsize*Xy) ,stepsize, lamb)

            cw = np.square(np.linalg.norm(np.dot(Xless,self.weights)-ytrain)) + lamb * np.linalg.norm(self.weights,ord = 1)
            iter += 1

    #Proximal operator
    def proximalOperator(self,prox,stepsize,lamb):

        for i,ele in enumerate(prox):
            if ele > stepsize * lamb:
                prox[i] -= stepsize * lamb
            elif np.abs(ele) <= stepsize * lamb:
                prox[i] = 0
            elif ele < stepsize * lamb:
                prox[i] += stepsize * lamb
        return prox


    def predict(self, Xtest):
        Xless = Xtest[:,self.params['features']]
        ytest = np.dot(Xless,self.weights)

        return ytest


class SGDLinearRegression(Regressor):
    def __init__(self,parameters):
        self.weights = 0
        self.params = {'features': [1, 2, 3]}
        self.reset(parameters)


    def reset(self, parameters):
        self.params = parameters

    def learn(self, Xtrain, ytrain):

        Xless = Xtrain[:, self.params['features']]
        numsamples = Xtrain.shape[0]

        stepsize = 0.01
        epochs = 500

        #init W
        self.weights = np.random.rand(Xless.shape[1])
        #start = time.time()
        for i in range(epochs):
            for j in range(numsamples):
                gcw = np.dot(( np.dot(Xless[j,:].T,self.weights)-ytrain[j] ),Xless[j,:] )

                self.weights -= stepsize*gcw
        #end= time.time()

        #rint('Time used to train SGD model in '+ str(epochs) +' epoches is: ' + str(start-end))

    def predict(self, Xtest):
        Xless = Xtest[:,self.params['features']]
        ytest = np.dot(Xless,self.weights)
        return ytest

class BatchGradientDescent(Regressor):
    def __init__(self,parameters):
        self.weights = 0
        self.params = {'features':[1,2,3]}
        self.reset(parameters)


    def reset(self, parameters):
        self.params = parameters

    def learn(self, Xtrain, ytrain):
        Xless = Xtrain[:, self.params['features']]
        numsamples = Xtrain.shape[0]

        #initialization
        error = 100000
        tolerance = 10e-4

        #init c(w)
        self.weights = np.random.rand(Xless.shape[1])
        cw = np.square(np.linalg.norm(np.dot(Xless,self.weights)-ytrain,ord=2))/(2*numsamples)

        #start loop
        maxIter = 3001
        iter = 0
        start = time.time()

        while (np.abs(cw-error) > tolerance) and (iter < maxIter):
            # if iter in [5, 50, 100, 300, 500, 1000, 3000, 5000, 8000, 10000]:
            #     end = time.time()
            #     print('time used for ' + str(iter) + ' is ' + str(end - start))

            error = cw
            gcw = np.dot(Xless.T, (np.dot(Xless, self.weights) - ytrain))/(numsamples)
            wt = self.weights

            #line-search
            stepsize = self.line_search(wt,Xless,ytrain)

            self.weights -= stepsize*gcw
            cw = np.square(np.linalg.norm(np.dot(Xless,self.weights)-ytrain))/(2*numsamples)
            iter += 1

    def line_search(self,wt,Xless,ytrain):

        #Optimization parameters
        gama = 0.5
        stepsize = 1.0
        tolerance = 10e-4

        #Original cw
        obj = np.square(np.linalg.norm(np.dot(Xless, wt) - ytrain))
        #cw after the first update
        w = wt - stepsize*np.dot(Xless.T, (np.dot(Xless, wt) - ytrain))
        c_w = np.square(np.linalg.norm(np.dot(Xless, w) - ytrain))

        #iter starts from 2nd update
        maxIter = 5000
        iter = 0
        while (obj - c_w < tolerance) and (iter <= maxIter):

            stepsize *= gama
            w = wt - stepsize * np.dot(Xless.T, (np.dot(Xless, wt) - ytrain))
            c_w = np.square(np.linalg.norm(np.dot(Xless, w) - ytrain))
            iter+=1

        return stepsize

    def predict(self, Xtest):
        Xless = Xtest[:, self.params['features']]
        ytest = np.dot(Xless, self.weights)
        return ytest



class AMSGRAD(Regressor):
    def __init__(self,parameters):
        self.weights = 0
        self.params = {'features': [1, 2, 3]}
        self.reset(parameters)

    def reset(self, parameters):
        self.params = parameters

    def learn(self, Xtrain, ytrain):
        Xless = Xtrain[:, self.params['features']]
        numsamples = Xtrain.shape[0]

        # init W
        self.weights = np.random.rand(Xless.shape[1])

        stepsize = 0.01
        epochs = 1000
        #initailization
        m = 0
        v = 0
        v_hat = np.zeros(384)
        bata1 = 0.1
        bata2 = 0.7

        for i in range(epochs):
            for j in range(numsamples):
                gcw = np.dot((np.dot(Xless[j, :].T, self.weights) - ytrain[j]), Xless[j, :])
                m = bata1*m + (1-bata2)* gcw
                v = bata2*v + (1-bata2) * np.square(gcw)

                for i, ele in enumerate(v):
                    if ele > v_hat[i]:
                        v_hat[i] = ele
                    else:
                        pass
                self.weights = self.weights - (stepsize/(np.sqrt(v_hat) + 0.001)) * m

    def predict(self, Xtest):
        Xless = Xtest[:,self.params['features']]
        ytest = np.dot(Xless,self.weights)
        return ytest






