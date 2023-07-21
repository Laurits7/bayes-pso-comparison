import os
import csv
import numpy as np
import pandas
import xgboost as xgb
from bayespso.tools import evaluation_tools as et
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


def create_dtrain(path_to_train):
    train_original_df = pandas.read_csv(path_to_train)
    train_df = train_original_df.copy()
    train_df['Label'] = train_original_df['Label'].replace(
        to_replace='s', value=1)
    train_df['Label'] = train_df['Label'].replace(
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
    training_labels = train_df['Label'].astype(int)
    traindataset = np.array(train_df[trainvars].values)
    weights = train_original_df['Weight'] * 550000/ len(training_labels)
    dtrain = xgb.DMatrix(
        traindataset,
        label=training_labels,
        nthread=8,
        feature_names=trainvars,
        missing=-999,
        weight=weights,
    )
    return dtrain, trainvars


def create_dtest(path_to_test, trainvars):
    test_df = pandas.read_csv(path_to_test)
    test_ids = test_df['EventId']
    dtest = xgb.DMatrix(
        test_df[trainvars],
        missing=-999,
        nthread=8
    )
    return dtest, test_ids


def create_submission(test_ids, pred_test, outfile, threshold_ratio=0.15):
    res  = [( int(test_ids[i]), pred_test[i] ) for i in range(len(pred_test))]
    rorder = {}
    for k, v in sorted(res, key = lambda x:-x[1]):
        rorder[ k ] = len(rorder) + 1
    ntop = int( threshold_ratio * len(rorder ) )
    fo = open(outfile, 'w')
    nhit = 0
    ntot = 0
    fo.write('EventId,RankOrder,Class\n')
    for k, v in res:
        if rorder[k] <= ntop:
            lb = 's'
            nhit += 1
        else:
            lb = 'b'
        fo.write('%s,%d,%s\n' % ( k,  len(rorder)+1-rorder[k], lb ) )
        ntot += 1
    fo.close()


def evaluate_test(model, path_to_test, trainvars, outfile, threshold):
    dtest, test_ids = create_dtest(path_to_test, trainvars)
    pred_test = model.predict(dtest)
    create_submission(test_ids, pred_test, outfile, threshold_ratio=threshold)


def submission_creation(path_to_train, path_to_test, hyperparameters, outfile, seed=1):
    dtrain, trainvars = create_dtrain(path_to_train)
    model = create_model(hyperparameters, dtrain, seed=seed)
    threshold = find_threshold(path_to_train, hyperparameters, seed=seed)
    thresholded_model = create_model(
        hyperparameters, dtrain, ams_threshold=threshold, seed=seed)
    print('Found threshold is: ' + str(threshold))
    evaluate_test(thresholded_model, path_to_test, trainvars, outfile, threshold)


def find_threshold(path_to_train, hyperparameters, seed=1):
    train_df, trainvars = load_df(path_to_train)
    threshold = kfold_threshold(train_df, trainvars, hyperparameters, seed=seed)
    return threshold


def kfold_threshold(prepared_data, trainvars, hyperparameters, seed=1):
    kfold = KFold(n_splits=5, shuffle=True, random_state=1)
    thresholds = []
    scores = []
    for train_index, test_index in kfold.split(prepared_data):
        train = prepared_data.iloc[train_index]
        test = prepared_data.iloc[test_index]
        dtrain, dtest = create_dmat(train, test, trainvars)
        threshold, score = evaluate(dtrain, dtest, hyperparameters, seed=seed)
        thresholds.append(threshold)
        scores.append(score)
    print('Scores mean: ' + str(np.mean(scores)))
    print('Scores std: ' + str(np.std(scores)))
    print('Thresholds: ' + str(thresholds))
    mean_threshold = np.mean(thresholds)
    return mean_threshold


def evaluate(dtrain, dtest, hyperparameters, seed=1):
    model = create_model(hyperparameters, dtrain, seed=seed)
    pred_train = model.predict(dtrain)
    pred_test = model.predict(dtest)
    threshold, score = calculate_test_threshold(
        pred_train,
        pred_test,
        dtrain,
        dtest,
        kappa=0.3,
    )
    return threshold, score


def create_model(hyperparameters, dtrain, ams_threshold=0.15, seed=1):
    ams_score_str = 'ams@' + str(ams_threshold)
    label = dtrain.get_label()
    weight = dtrain.get_weight()
    sum_wpos = sum(weight[i] for i in range(len(label)) if label[i] == 1.0)
    sum_wneg = sum(weight[i] for i in range(len(label)) if label[i] == 0.0)
    parameters = {
        'objective': 'binary:logitraw',
        'scale_pos_weight': sum_wneg/sum_wpos,
        'eval_metric': 'auc',
        'silent': 1,
        'seed': seed
    }
    watchlist = [(dtrain,'train')]
    hyp_copy = hyperparameters.copy()
    num_boost_round = hyp_copy.pop('num_boost_round')
    parameters.update(hyp_copy)
    parameters = list(parameters.items())+[('eval_metric', ams_score_str)]
    model = xgb.train(
        parameters,
        dtrain,
        num_boost_round,
        watchlist
    )
    return model


def create_dmat(train, test, trainvars):
    dtrain = xgb.DMatrix(
        train[trainvars].values,
        label=np.array(train['Label'], dtype=int),
        nthread=8,
        feature_names=trainvars,
        missing=-999,
        weight=np.array(train['Weight'], dtype=float),
    )
    dtest = xgb.DMatrix(
        test[trainvars],
        label=np.array(test['Label'], dtype=int),
        nthread=8,
        feature_names=trainvars,
        missing=-999,
        weight=np.array(test['Weight'], dtype=float),
    )
    return dtrain, dtest


def calculate_test_threshold(
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
    test_threshold, best_ams_score = try_different_thresholds(
        pred_test, test_weights, test_labels, factor='test')
    # train_threshold = try_different_thresholds(pred_train, train_weights, train_labels, factor='train')
    return test_threshold, best_ams_score


def try_different_thresholds(predicted, weights, labels, factor):
    thresholds = np.arange(0, 1, 0.005)
    ams_scores = []
    theta_cuts = []
    for threshold in thresholds:
        ams_score, theta_cut = single_threshold(predicted, weights, labels, threshold, factor)
        ams_scores.append(ams_score)
        theta_cuts.append(theta_cut)
    index = np.argmax(ams_scores)
    the_threshold = theta_cuts[index]
    best_ams_score = ams_scores[index]
    print(factor + ' best score: ' + str(best_ams_score))
    print(str(thresholds[index] * 100) + '% is classified as the signal for the best cut')
    print('###################################################################')
    return thresholds[index], best_ams_score


def single_threshold(predicted, weights, labels, threshold, factor):
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
    labels = dmat.get_label()
    weights = np.array(dmat.get_weight())*factor
    signal, background = calculate_s_and_b(prediction, labels, weights)
    ams_score = ams(signal, background)
    return ams_score


def ams(s, b):
    """ Approximate Median Significance defined as:
        AMS = sqrt(
                2 { (s + b + b_r) log[1 + (s/(b+b_r))] - s}
              )        
    where b_r = 10, b = background, s = signal, log is natural logarithm
    """
    br = 10.0
    radicand = 2 *( (s+b+br) * np.log (1.0 + s/(b+br)) -s)
    if radicand < 0:
        print('radicand is negative. Exiting')
        exit()
    else:
        return np.sqrt(radicand)

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