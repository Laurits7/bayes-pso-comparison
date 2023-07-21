import pandas
import numpy as np
import xgboost as xgb
from bayespso.tools import evaluation_tools as et
from sklearn.model_selection import KFold


def load_df(path_to_train):
    train_original_df = pandas.read_csv(path_to_train)
    train_df = train_original_df.copy()
    train_original_df['Label'] = train_original_df['Label'].replace(
        to_replace='s', value=1)
    train_original_df['Label'] = train_original_df['Label'].replace(
        to_replace='b', value=0)
    labels_to_drop = ['Kaggle', 'EventId', 'Weight']
    for trainvar in train_df.columns:
        for label_to_drop in labels_to_drop:
            if label_to_drop in trainvar:
                try:
                    train_df = train_df.drop(trainvar, axis=1)
                except:
                    continue
    trainvars = list(train_df.columns)
    trainvars.remove('Label')
    return train_original_df, trainvars


def kfold(prepared_data, trainvars, hyperparameters):
    kfold = KFold(n_splits=5, shuffle=True, random_state=1)
    ams_scores = []
    test_amss =[]
    train_amss = []
    for train_index, test_index in kfold.split(prepared_data):
        train = prepared_data.iloc[train_index]
        test = prepared_data.iloc[test_index]
        dtrain, dtest = create_dmat(train, test, trainvars)
        d_ams, test_ams, train_ams = evaluate(dtrain, dtest, hyperparameters)
        ams_scores.append(d_ams)
        test_amss.append(test_ams)
        train_amss.append(train_ams)
        del dtrain
        del dtest
        del train
        del test
    return ams_scores, test_amss, train_amss


def evaluate(dtrain, dtest, hyperparameters):
    model = create_model(hyperparameters, dtrain)
    pred_train = model.predict(dtrain)
    pred_test = model.predict(dtest)
    d_ams, test_ams, train_ams = calculate_d_ams(
        pred_train,
        pred_test,
        dtrain,
        dtest,
        kappa=0.3,
    )
    return d_ams, test_ams, train_ams


def create_dmat(train, test, trainvars):
    dtrain = xgb.DMatrix(
        train[trainvars].values,
        label=np.array(train['Label'], dtype=int),
        nthread=4,
        feature_names=trainvars,
        missing=-999,
        weight=np.array(train['Weight'], dtype=float),
    )
    dtest = xgb.DMatrix(
        test[trainvars],
        label=np.array(test['Label'], dtype=int),
        nthread=4,
        feature_names=trainvars,
        missing=-999,
        weight=np.array(test['Weight'], dtype=float),
    )
    return dtrain, dtest


def create_model(hyperparameters, dtrain):
    label = dtrain.get_label()
    weight = dtrain.get_weight()
    sum_wpos = sum(weight[i] for i in range(len(label)) if label[i] == 1.0)
    sum_wneg = sum(weight[i] for i in range(len(label)) if label[i] == 0.0)
    parameters = {
        'objective': 'binary:logitraw',
        'scale_pos_weight': sum_wneg/sum_wpos,
        'eval_metric': 'auc',
        'nthread': 4,
        'silent': 1
    }
    watchlist = [(dtrain,'train')]
    hyp_copy = hyperparameters.copy()
    num_boost_round = hyp_copy.pop('num_boost_round')
    parameters.update(hyp_copy)
    parameters = list(parameters.items())+[('eval_metric', 'ams@0.15')]
    model = xgb.train(
        parameters,
        dtrain,
        num_boost_round,
        watchlist
    )
    return model


def create_dtest(path_to_test, trainvars):
    test_df = pandas.read_csv(path_to_test)
    test_ids = test_df['EventId']
    dtest = xgb.DMatrix(
        test_df[trainvars],
        missing=-999,
        nthread=4
    )
    return dtest, test_ids


def calculate_s_and_b(prediction, labels, weights):
    signal = 0
    background = 0
    prediction = np.array(prediction)
    labels = np.array(labels)
    weights = np.array(weights)
    for i in range(len(prediction)):
        if prediction[i] == 1:
            if labels[i] == 1:
                signal += weights[i]
            elif labels[i] == 0:
                background += weights[i]
    return signal, background


def calculate_d_ams(
        pred_train,
        pred_test,
        dtrain,
        dtest,
        kappa=0.3,
):
    test_weights = np.array(dtest.get_weight())*5
    train_weights = np.array(dtrain.get_weight())*1.25
    train_labels = dtrain.get_label()
    test_labels = dtest.get_label()
    test_threshold = try_different_thresholds(
        pred_test, test_weights, test_labels, factor='test')
    train_threshold = try_different_thresholds(
        pred_train, train_weights, train_labels, factor='train')
    pred_threshold = test_threshold
    print("Test threshold: " + str(test_threshold))
    print("Train threshold: " + str(train_threshold))
    classified_train = np.array(
        [1 if prediction >= pred_threshold else 0 for prediction in pred_train])
    classified_test = np.array(
        [1 if prediction >= pred_threshold else 0 for prediction in pred_test])
    train_ams = calculate_ams(dtrain, classified_train, factor=1.25)
    test_ams = calculate_ams(dtest, classified_test, factor=5)
    d_ams = et.calculate_d_score(train_ams, test_ams, kappa)
    print("train AMS: " + str(train_ams))
    print("test AMS: " + str(test_ams))
    print("d_ams: " + str(d_ams))
    return d_ams, test_ams, train_ams


def try_different_thresholds(predicted, weights, labels, factor):
    thresholds = np.arange(0, 1, 0.005)
    ams_scores = []
    theta_cuts = []
    for threshold in thresholds:
        ams_score, theta_cut = single_threshold(
            predicted, weights, labels, threshold)
        ams_scores.append(ams_score)
        theta_cuts.append(theta_cut)
    index = np.argmax(ams_scores)
    the_threshold = theta_cuts[index]
    best_ams_score = ams_scores[index]
    print(factor + ' best score: ' + str(best_ams_score))
    print(
        str(thresholds[index] * 100)\
        + '% is classified as the signal for the best cut'
    )
    print('###################################')
    return the_threshold


def single_threshold(predicted, weights, labels, threshold):
    ''' Computes the AMS score for a given threshold

    Parameters:
    ----------
    predicted : 

    weights : list
        list of the weights of each event
    labels : list
        list of labels for each event
    threshold : float
        Threshold starting from which a prediction is classified as signal

    Returns:
    -------
    ams_score : float
        AMS score corresponding to a given threshold and prediction
    theta_cut : 
    '''
    number_signal = int(np.ceil(len(predicted)* threshold))
    pr = list(predicted)
    theta_cut = min(sorted(pr)[-number_signal:])
    prediction = [1 if pred >= theta_cut else 0 for pred in predicted]
    signal, background = calculate_s_and_b( 
        prediction, labels, weights)
    ams_score = ams(signal, background)
    return ams_score, theta_cut



def calculate_s_and_b(prediction, labels, weights):
    '''Calculates amount of signal and background. When given weights, possible
    to have weighed signal and background

    Parameters:
    ----------
    prediction : list
        Prediction for each event. (list of int)
    labels : list / pandas Series
        True label for each event
    weights : list
        list of floats. Weight for each event
    [weighed=True] : bool
        Whether to use the weights for calculating singal and background

    Returns:
    -------
    signal : int (float)
        Number of (weighed) signal events in the ones classified as signal
    background : int
        Number of (weighed) background events in the ones classified as signal
    '''
    signal = 0
    background = 0
    prediction = np.array(prediction)
    labels = np.array(labels)
    weights = np.array(weights)
    for i in range(len(prediction)):
        if prediction[i] == 1:
            if labels[i] == 1:
                signal += weights[i]
            elif labels[i] == 0:
                background += weights[i]
    return signal, background


def calculate_ams(dmat, prediction, factor):
    '''Calculates the AMS score from the prediction

    Parameters:
    ----------
    dmat : xgb.DMatrix
        The DMatrix created from given data, weights and labels
    prediction : list
        Predicted labels for the data
    factor : float
        The factor all the weights should be multiplied in order to get the
        correct AMS score (due to the splitting of training data)

    Returns:
    -------
    ams_score : flaot
        AMS score calculated based on the prediction and the given labels and
        weights.
    '''
    labels = dmat.get_label()
    weights = np.array(dmat.get_weight())*factor
    signal, background = calculate_s_and_b(prediction, labels, weights)
    ams_score = ams(signal, background)
    return ams_score


def ams(s, b):
    ''' 
    Approximate Median Significance defined as:
        AMS = sqrt(
                2 { (s + b + b_r) log[1 + (s/(b+b_r))] - s}
              )        
    where b_r = 10, b = background, s = signal, log is natural logarithm
    '''
    br = 10.0
    radicand = 2 *( (s+b+br) * np.log (1.0 + s/(b+br)) -s)
    if radicand < 0:
        print('Radicand is negative. Exiting')
        exit()
    else:
        return np.sqrt(radicand)


def higgs_evaluation_main(path_to_train, hyperparameters):
    ''' Evaluates the ATLAS Higgs challenge training data

    Parameters:
    ----------
    path_to_train : str
        Path to the train dataset
    hyperparameters: dict
        Hyperparameters to be used in the model building

    Returns:
    -------
    ams_scores : list
        The d-AMS scores from KFold CV
    test_amss : list
        AMS scores of the test samples from KFold CV
    train_amss : list
        AMS scores of the train samples from KFold CV
    '''
    train_df, trainvars = load_df(path_to_train)
    ams_scores, test_amss, train_amss = kfold(train_df, trainvars, hyperparameters)
    return ams_scores, test_amss, train_amss


def ensemble_fitness(hyperparameter_sets, settings):
    scores = []
    for hyperparameters in hyperparameter_sets:
        score = higgs_evaluation_main(
            settings['train_file'], hyperparameters)[0]
        scores.append(score)
    return scores