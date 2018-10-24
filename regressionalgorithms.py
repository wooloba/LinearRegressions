from __future__ import division  # floating point division
import numpy as np
import math

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
        self.weights = np.dot(np.dot(np.linalg.inv(np.dot(Xless.T,Xless)/numsamples), Xless.T),ytrain)/numsamples

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
    def __init__( self,lam, parameters={}):
        # Default parameters, any of which can be overwritten by values passed to params
        self.weights = None
        self.lamb =  lam
        self.params = {'features': [1, 2, 3, 4, 5]}
        self.reset(parameters)

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
    def __init__(self,parameters):
        self.weights = 0
        self.params = {'features':[1,2,3]}
        self.reset(parameters)
        self.lamb = 0.01

        self.error = 10000
        self.tolerance = 10e-4


    def learn(self, Xtrain, ytrain):
        Xless = Xtrain[:, self.params['features']]
        numsamples = Xtrain.shape[0]

        #precomputing
        XX = np.dot(np.dot(Xless.T,Xless),1/numsamples)
        Xy = np.dot(np.dot(Xless.T,ytrain),1/numsamples)

        #stepsize
        stepsize = 1/(2*np.linalg.norm(XX))

        #initial c(w)
        cw = np.square(np.linalg.norm(np.dot(Xless,self.weights)-ytrain,ord=2)) + self.lamb * np.linalg.norm(self.weights,ord = 1)

        #define max iteration
        iter = 0
        maxIter = 5000
        while np.abs(np.subtract(cw , self.error)) > self.tolerance and (iter < maxIter):
            self.error = cw
            self.weights = self.proximalOperator( (self.weights - stepsize*XX*self.weights+stepsize*Xy) ,stepsize)

            iter += 1

    #Proximal operator
    def proximalOperator(self,prox,stepsize):
        for ele,i in enumerate(prox):
            if ele > stepsize*self.lamb:
                prox[i] -= stepsize*self.lamb
            elif np.abs(ele) <= stepsize*self.lamb:
                prox[i] = 0
            elif ele < stepsize*self.lamb:
                prox[i] += stepsize*self.lamb
        return prox


    def predict(self, Xtest):
        Xless = Xtest[:,self.params['features']]
        return np.dot(Xless,self.weights)





















