import xgboost as xgb
# from xgboost import plot_importance
import pandas as pd
import numpy as np
import random
import time
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import pickle
import os


def build_model():
    # we include hyperparameter search into xgboost model, and treating it as the total estimating model
    # use randomized grid search to save time
    params_dic = {
        'silent': [1],
        'learning_rate': [0.001, 0.01, 0.1, 0.2, 0,3],
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
    random_search_model = RandomizedSearchCV(xgb_model, param_distributions=params_dic, n_iter=200, cv=3, verbose=0, n_jobs=-1)
    return random_search_model

def model_predict(data_train, x_forecast, saving_path, n_iterations = 1000, confidence=95):

    # bootstrap: select subsamples to generate predictions by the 'model',
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
    model_ls = []

    for i in range(n_iterations):

        # randomly choose sample size>=30 from the full data set,
        # and split into train set and test set by randomly choose a ratio between 0.1 to 0.5 for test set
        sample_size = random.randint(20, data_train.shape[0])
        sample = data_train.sample(sample_size)
        test_ratio = random.uniform(0.1, 0.5)
        x_train, x_test, y_train, y_test = train_test_split(sample[x_label], sample[y_label], test_size=test_ratio)
        # shuffle = True

        training_time_start = time.time()
        model = build_model()
        model.fit(x_train, y_train)
        training_time = time.time() - training_time_start
        model_ls.append(model)

        predict = model.predict(x_test)
        iterate_col_str = 'iterate_%d' %i
        prediction = pd.DataFrame(predict, index = y_test.index, columns= [iterate_col_str])
        prediction_result = pd.concat([prediction_result, prediction], axis=1)
        fitness.loc[iterate_col_str, 'mse'] = mean_squared_error(y_test, predict)
        fitness.loc[iterate_col_str, 'training_time'] = training_time
        fitness.loc[iterate_col_str, 'sample_size'] = sample_size
        fitness.loc[iterate_col_str, 'estimators'] = model.best_estimator_

    ## evaluations
    # confidence interval
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
    prediction_result['Predictive Power'] = predictive_power
    #print(prediction_result)
    print('At %d confidence, the prediction score is ' % confidence, predictive_power)
    prediction_result.to_csv(saving_path+'/prediction_results.csv')

    # find smallest mse and use according model to forecast
    fitness_min = fitness[['mse','training_time']].min()
    fitness_idxmin = fitness[['mse','training_time']].idxmin()
    print('the smallest mse is', fitness_min['mse'], ', training took', fitness.loc[fitness_idxmin['mse'],
                                                                                    'training_time'],
          's, from', fitness.loc[fitness_idxmin['mse'],'sample_size'], 'samples.')
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

    return forecast[0], predictive_power
