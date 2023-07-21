import pandas
import numpy as np
import time
import xgboost as xgb

PATH_TO_DATA = '/home/user/atlas-higgs-challenge-2014-v2.csv'
NR_TRAIN = 250000
NR_TEST = 550000
HYPERPARAMETERS = {
    "num_boost_round": 700,
    "subsample": 0.8,
    "colsample_bytree": 0.7644932408891923,
    "gamma": 0.060799740781140386,
    "learning_rate": 0.01,
    "max_depth": 7,
    "min_child_weight": 163.4093476157902
} # just some random variables

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


def create_model(hyperparameters, dtrain):
    label = dtrain.get_label()
    weight = dtrain.get_weight()
    sum_wpos = sum(weight[i] for i in range(len(label)) if label[i] == 1.0)
    sum_wneg = sum(weight[i] for i in range(len(label)) if label[i] == 0.0)
    parameters = {
        'objective': 'binary:logitraw',
        'scale_pos_weight': sum_wneg/sum_wpos,
        'eval_metric': 'auc',
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


def evaluate(dtrain, dtest, hyperparameters):
    model = create_model(hyperparameters, dtrain)
    pred_train = model.predict(dtrain)
    pred_test = model.predict(dtest)
    # d_ams, test_ams, train_ams = calculate_d_ams(
    #     pred_train,
    #     pred_test,
    #     dtrain,
    #     dtest,
    #     kappa=0.3,
    # )
    d_ams, test_ams, train_ams = 1, 2, 3
    return d_ams, test_ams, train_ams


def main():
    data, trainvars = load_df(PATH_TO_DATA)
    train = data.iloc[:NR_TRAIN]
    test = data.iloc[NR_TRAIN:NR_TRAIN + NR_TEST]
    dtrain, dtest = create_dmat(train, test, trainvars)
    start = time.time()
    d_ams, test_ams, train_ams = evaluate(dtrain, dtest, HYPERPARAMETERS)
    end = time.time()
    print('Duration: %ss' %(end-start))


if __name__ == '__main__':
    main()
