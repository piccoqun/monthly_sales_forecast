import xgboost as xgb
# from xgboost import plot_importance
import pandas as pd
import numpy as np
import time
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from plot import plot_histogram
import pickle
import os

def build_model(target_mean):
    # we include hyperparameter search into xgboost model, and treating it as a whole training model
    # use randomized grid search to save time
    params_dic = {
        'silent': [1], # not showing running messages
        # learning task parameters
        'learning_rate': [0.001, 0.01, 0.1, 0.2, 0,3],
        'base_score':[target_mean], # initialize prediction score (global bias) to make sure the trees 'catching up' faster
        # tree based parameters
        'max_depth': [3, 6, 10, 15, 20],
        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
        'gamma': [0, 0.25, 0.5, 1.0],
        # regularization parameters
        'alpha': [0.0001, 0.001, 0.1, 1.0, 5.0, 10.0, 50.0],
    }
    xgb_model = xgb.XGBRegressor()
    random_search_model = RandomizedSearchCV(xgb_model, param_distributions=params_dic, n_iter=200, cv=3, verbose=0,
                                             iid=False, n_jobs=-1)
    # param_distributions: if a list is given, sample uniformly
    # n_iter: not all the combinations are tested, n_iter set the number of total combinations,
    # the more the better prediction but slower running time
    # n_jobs: try as many processors as possible
    # iid: return the average score across folds, not weighted by the number of samples in each test set
    # cv: KFold cross validation
    return random_search_model

def model_predict(data_train, x_forecast, saving_path, n_iterations = 1000, confidence=95):

    # bootstrap: select subsamples to generate predictions,
    # hence to build up prediction confidence intervals

    if not os.path.exists(saving_path):
        os.mkdir(saving_path)
    '''
        with open(saving_path + '/model.sav', 'rb') as file1:
            best_model = pickle.load(file1)
        # there is problem to predict by reading saved model: feature names are mismatched
        prediction_result = pd.read_csv(saving_path+'/prediction_results.csv')
        predictive_power = prediction_result['Predictive Power'].iloc[0]
    '''

    x_label = x_forecast.columns
    y_label = ['Actuals']
    fitness = pd.DataFrame()
    prediction_result = pd.DataFrame(index = data_train.index)
    evaluation = {}
    model_ls = []

    data_train[y_label] = data_train[y_label]

    for i in range(n_iterations):

        # bootstrap: resample 80% of total data to train, and test on the unused data for performance - mse
        # this loop results in two dataframe: one for predictions and one for fitness
        # empirically the training sample ratio have influence on the predictive power: 0.6 > 0.7 >0.8
        sample_size = int(data_train.shape[0]*0.7)
        # resample shuffles data
        train_sample = resample(data_train, n_samples=sample_size, replace=False)
        test_sample = data_train[~data_train.index.isin(train_sample.index)]
        x_train = train_sample[x_label]
        y_train = train_sample[y_label]
        label_mean = y_train.mean().values[0]
        x_test = test_sample[x_label]
        y_test = test_sample[y_label]

        training_time_start = time.time()
        model = build_model(label_mean)
        model.fit(x_train, y_train)
        training_time = time.time() - training_time_start
        model_ls.append(model)

        predict = model.predict(x_test)
        iterate_col_str = 'iterate_%d' %i
        prediction = pd.DataFrame(predict, index = y_test.index, columns= [iterate_col_str])
        prediction_result = pd.concat([prediction_result, prediction], axis=1)
        fitness.loc[iterate_col_str, 'mse'] = mean_squared_error(y_test, predict)
        fitness.loc[iterate_col_str, 'training_time'] = training_time
        fitness.loc[iterate_col_str, 'estimators'] = model.best_estimator_

    ## evaluations

    # confidence interval of prediction results
    for index in prediction_result.index:
        prediction_row = prediction_result.loc[index].dropna()
        if prediction_row.empty:
            continue
        CI_low = np.percentile(prediction_row, [(100 - confidence)/2.])
        CI_up = np.percentile(prediction_row, [100 - (100 - confidence)/2.])
        prediction_result.loc[index, 'CI_up'] = CI_up
        prediction_result.loc[index, 'CI_low'] = CI_low

    # fill nan with mean value in CI_up and CI_low, so that it can be compared later
    prediction_result[['CI_up', 'CI_low']] = prediction_result[['CI_up','CI_low']].fillna(prediction_result[['CI_up','CI_low']].mean())
    prediction_result['In_CI'] = np.where((data_train[y_label].values<prediction_result[['CI_up']].values) &
                                          (data_train[y_label].values>prediction_result[['CI_low']].values), 1, 0)

    predictive_power = prediction_result['In_CI'].sum() / prediction_result.shape[0]
    prediction_result.to_csv(saving_path + '/prediction_results.csv')
    evaluation['Predictive Power'] = predictive_power
    #print(prediction_result)
    print('At %d confidence, the prediction score is ' % confidence, predictive_power)

    # confidence interval of mse
    plot_histogram(fitness[['mse']], folder=saving_path, title='mse_hit')
    mse_CI_low = np.percentile(fitness['mse'].values, [(100 - confidence)/2.])
    mse_CI_up = np.percentile(fitness['mse'].values, [100 - (100 - confidence)/2.])
    evaluation['MSE CI Lower Bound'] = mse_CI_low
    evaluation['MSE CI Upper Bound'] = mse_CI_up
    print('{}% confidence of mse is {} and {}'.format(confidence, mse_CI_low, mse_CI_up))

    with open(saving_path +'/evaluation_dictionary.pkl', 'wb') as dict_file:
        pickle.dump(evaluation, dict_file, protocol=pickle.HIGHEST_PROTOCOL)

    # find smallest mse and use according model to forecast
    fitness_min = fitness[['mse','training_time']].min()
    fitness_idxmin = fitness[['mse','training_time']].idxmin()
    print('the smallest mse is', fitness_min['mse'], ', training took', fitness.loc[fitness_idxmin['mse'],
                                                                                    'training_time'],'s.')
    fitness.to_csv(saving_path+'/fitness.csv')

    # print('we choose the estimator with smallest validation mse.')
    best_model = model_ls[fitness.index.get_loc(fitness_idxmin['mse'])]
    with open(saving_path+'/model.sav', 'wb') as file:
        pickle.dump(best_model, file, protocol=pickle.HIGHEST_PROTOCOL)

    ''''
    # only applies to Booster models
    plot_importance(best_model)
    plt.savefig(saving_path+'/feature_importance.png')
    plt.show()
    '''

    forecast = best_model.predict(x_forecast)

    return forecast[0], evaluation
