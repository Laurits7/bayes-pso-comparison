import numpy as np


def rosenbrock_function(
        parameter_dict,
        a=1,
        b=100
):
    ''' The Rosenbrock function

    Parameters:
    -----------
    parameter_dict : dict
        Dictionary containing the hyperparameters (the coordinates of the
        point to be evaluated)
    [a=1] : float
        The parameter 'a' of the Rosenbrock function
    [b=100] : float
        The parameter 'b' of the Roisenbrock function

    Returns:
    -------
    score : float
        The function valueat the coordinates 'x' and 'y'. Returns the negative
        Rosenbrock function value.
    '''
    score = (
        (a - parameter_dict['x'])**2
        + b*(parameter_dict['y']- parameter_dict['x']**2)**2
    )
    return score


def ensemble_rosenbrock(
        parameter_dicts,
        true_values={'a': 1, 'b': 100}
):
    ''' Calcualtes the Rosenbrock function value for the ensemble.

    Parameters:
    -----------
    parameter_dicts : list of dicts
        List of the coordinate dictionaries
    true_values : dict
        Dummy

    Returns:
    --------
    scores : list
        Scores for each member in the ensemble
    '''
    scores = []
    for parameter_dict in parameter_dicts:
        score = rosenbrock_function(
            parameter_dict,
            1,
            100)
        scores.append(score)
    return scores


def branin_hoo(parameter_dict):
    x = parameter_dict['x']
    y = parameter_dict['y']
    score = (
        (y - 5.1* x**2 / (4*np.pi**2) + 5*x/np.pi - 6)**2 + 10*(1 - 1/(8*np.pi)) * np.cos(x) + 10
    )
    return score
