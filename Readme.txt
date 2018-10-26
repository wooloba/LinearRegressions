    To disable/enable algorithm you want to test just comment/uncomment out the line of algorithm
    parameters in each algorithm will be replaced. In default, all algorithm will be turned on.
    And there are 3 runs in default. You can change the run number at line 32 of regressionalgorithm.py.


    Parameters in each algorithm will be replaced by following variable.
    Only Lasso used regwet to choose the best lambda. In my code, however, the
    choosing part has been disabled.
    To enable choose the best lambda, simply uncomment Line 64/66 in regressionalgorithm.py.

    Other algorithms only need features variable.

    For FSLinearRegression, using large feature value would crash since matrix is not full rank and
    cannot be inversed. I have handled this problem by replacing
    np.linalg.inv to np.linalg.pinv (Line 103 of regressionalgorithm.py).
    pinv method will calculatepersudo inverse of a matrix.
    Therefore, problem caused by large number of features will be handled in this case.


    Data is not included in the zip file.