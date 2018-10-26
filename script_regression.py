from __future__ import division  # floating point division
import csv
import random
import math
import numpy as np

import dataloader as dtl
import regressionalgorithms as algs

def l2err(prediction,ytest):
    """ l2 error (i.e., root-mean-squared-error) """
    return np.linalg.norm(np.subtract(prediction,ytest))

def l1err(prediction,ytest):
    """ l1 error """
    return np.linalg.norm(np.subtract(prediction,ytest),ord=1)

def l2err_squared(prediction,ytest):
    """ l2 error squared """
    return np.square(np.linalg.norm(np.subtract(prediction,ytest)))

def geterror(predictions, ytest):
    # Can change this to other error values
    return l2err(predictions,ytest)/ytest.shape[0]


if __name__ == '__main__':
    trainsize = 1000
    testsize = 5000

    #Here, you can increase or decrease the number of runs.
    numruns = 3


    '''
    To disable/enable algorithm you want to test just comment/uncomment out the line of algorithm
    parameters in each algorithm will be replaced. See comments below.
    '''
    regressionalgs = {
                'Random': algs.Regressor(),
                'Mean': algs.MeanPredictor(),
                'FSLinearRegression5': algs.FSLinearRegression({'features': range(5)}),
                'FSLinearRegression50': algs.FSLinearRegression({'features': range(50)}),
                'RidgeLinearRegression': algs.RidgeLinearRegression(parameters={'features': range(384),'lamb':0.01}),
                'LassoLinearRegression': algs.LassoLinearRegression(parameters={'regwgt': None,'features':range(1)}),
                'SGDLinearRegression':algs.SGDLinearRegression(parameters={'regwgt': None,'features':range(1)}),
                'BatchGradientDescent':algs.BatchGradientDescent(parameters={'regwgt': None,'features':range(1)}),
                'AMSGRAD':algs.AMSGRAD(parameters={'regwgt': None,'features':range(1)})
             }
    numalgs = len(regressionalgs)

    # Enable the best parameter to be selected, to enable comparison
    # between algorithms with their best parameter settings
    '''
    Parameters in each algorithm will be replaced by following variable.
    Only Lasso used regwet. Other algorithm only need features variable.
    For FSLinearRegression, using large feature value would crash since matrix is not full rank and 
    cannot be inversed. I have handled this problem by replacing 
    np.linalg.inv to np.linalg.pinv (Line 103 of regressionalgorithm.py). pinv method will calculate
    persudo inverse of a matrix. 
    Therefore, problem caused by large number of features will be handled in this case. 
    '''
    parameters = (
        #{'regwgt': 0.0,'features':range(384)},
        {'regwgt': 0.01,'features':range(384)},
        #{'regwgt': 1.0,'features':range(384)},
                      )

    numparams = len(parameters)
    
    errors = {}
    for learnername in regressionalgs:
        errors[learnername] = np.zeros((numparams,numruns))

    for r in range(numruns):
        print("this is "+str(r)+" run.")
        trainset, testset = dtl.load_ctscan(trainsize,testsize)
        print(('Running on train={0} and test={1} samples for run {2}').format(trainset[0].shape[0], testset[0].shape[0],r))

        for p in range(numparams):
            params = parameters[p]
            for learnername, learner in regressionalgs.items():
                # Reset learner for new parameters
                learner.reset(params)
                print ('Running learner = ' + learnername + ' on parameters ' + str(learner.getparams()))
                # Train model
                learner.learn(trainset[0], trainset[1])
                # Test model
                predictions = learner.predict(testset[0])
                error = geterror(testset[1], predictions)

                print ('Error for ' + learnername + ': ' + str(error))
                print('\n')
                errors[learnername][p,r] = error
        #print(errors)

    for learnername in regressionalgs:
        besterror = np.mean(errors[learnername][0,:])
        bestparams = 0
        for p in range(numparams):
            aveerror = np.mean(errors[learnername][p,:])

            sdError = np.std(errors[learnername][p,:])/np.sqrt(numruns)
            if aveerror < besterror:
                besterror = aveerror
                bestparams = p

        # Extract best parameters
        learner.reset(parameters[bestparams])
        #print ('Best parameters for ' + learnername + ': ' + str(learner.getparams()))
        print ('Average error for ' + learnername + ': ' + str(besterror))
        print ('Standard error for ' + learnername + ': ' + str(sdError))
        print('\n')