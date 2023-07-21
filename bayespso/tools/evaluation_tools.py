'''Tools to be used for evaluation of the model'''
import numpy as np
import pandas


def calculate_d_score(train_score, test_score, kappa=0.3):
    ''' Calculates the d_score with the given kappa, train_score and
    test_score. Can be used to get d_auc, d_ams or other similar.

    Parameters:
    ----------
    train_score : float
        Score of the training sample
    test_score : float
        Score of the testing sample
    kappa : float
        Weighing factor for the difference between test and train auc

    Returns:
    -------
    d_roc : float
        Score based on D-score and AUC
    '''
    difference = abs(train_score - test_score)
    d_score = test_score - kappa * difference
    return d_score


def calculate_d_ams(
        pred_train,
        pred_test,
        data_dict,
        kappa=0.3
):
    '''Calculates the d_ams score

    Parameters:
    ----------
    pred_train : list of lists
        Predicted labels of the training dataset
    pred_test : list of lists
        Predicted labels of the testing dataset
    data_dict : dict
        Dictionary that contains the labels for testing and training. Keys are
        called 'testing_labels' and 'training_labels'

    Returns:
    -------
    d_ams : float
        the ams score calculated using the d_score function
    '''
    train_ams, best_threshold = try_different_thresholds(
        pred_train, data_dict, 'train')
    test_ams = try_different_thresholds(
        pred_test, data_dict, 'test', threshold=best_threshold)
    d_ams = calculate_d_score(train_ams, test_ams, kappa)
    return d_ams


def calculate_compactness(parameter_dicts):
    '''Calculates the improvement based on how similar are different sets of
    parameters

    Parameters:
    ----------
    parameter_dicts : list of dicts
        List of dictionaries to be compared for compactness.

    Returns:
    -------
    mean_cov : float
        Coefficient of variation of different sets of parameters.
    '''
    keys = parameter_dicts[0].keys()
    list_dict = values_to_list_dict(keys, parameter_dicts)
    mean_cov = calculate_dict_mean_coeff_of_variation(list_dict)
    return mean_cov


def values_to_list_dict(keys, parameter_dicts):
    '''Adds same key values from different dictionaries into a list w

    Parameters:
    ----------
    keys : list
        list of keys for which same key values are added to a list
    parameter_dicts : list of dicts
        List of parameter dictionaries.

    Returns:
    -------
    list_dict: dict
        Dictionary containing lists as valus.
    '''
    list_dict = {}
    for key in keys:
        key = str(key)
        list_dict[key] = []
        for parameter_dict in parameter_dicts:
            list_dict[key].append(parameter_dict[key])
    return list_dict


def calculate_dict_mean_coeff_of_variation(list_dict):
    '''Calculate the mean coefficient of variation for a given dict filled
    with lists as values

    Parameters:
    ----------
    list_dict : dict
        Dictionary containing lists as values

    Returns:
    -------
    mean_coeff_of_variation : float
        Mean coefficient of variation for a given dictionary haveing lists as
        values
    '''
    coeff_of_variations = []
    for key in list_dict:
        values = list_dict[key]
        coeff_of_variation = np.std(values)/abs(np.mean(values))
        coeff_of_variations.append(coeff_of_variation)
    mean_coeff_of_variation = np.mean(coeff_of_variations)
    return mean_coeff_of_variation
