import pandas as pd
from datetime import datetime

def main():
    # step 1: load data
    from data_generator import load_data
    xy_train_df, forecast_dict = load_data()

    # step 2: obtain visual and statistical reports
    from plot import plot_df, plot_histogram
    plot_df(df = xy_train_df, title='history monthly data', subplots=True)
    plot_histogram(df= xy_train_df, title = 'history monthly histogram')

    # step 3: train model and make the forecast
    from xgboost_model import model_predict
    y_label = ['Actuals']
    x_label = list(set(xy_train_df.columns) - set(y_label))
    forecast_result = pd.DataFrame(columns=['Forecast','Predictive Power','Running Time'])
    for forecast_date in forecast_dict.keys():
        forecast_date_str = forecast_date.strftime('%d/%m/%Y')
        print('start forecasting for ' + forecast_date_str)
        forecast_start_time = datetime.now()
        saving_path = 'reports/model_'+ forecast_date_str
        x_forecast = forecast_dict[forecast_date].iloc[[-1]][x_label]
        # default n_iterations=1000, confidence=95
        n_iterations = 5
        forecast_value, predict_power = model_predict(data_train = xy_train_df, x_forecast=x_forecast,
                                                      saving_path=saving_path, n_iterations=n_iterations)
        time_delta = str(datetime.now() - forecast_start_time)
        forecast_result.loc[forecast_date] = [forecast_value, predict_power, time_delta]

        print('forecasting for {} is finished, took {}s for {} iterations'.format(forecast_date_str,
                                                                                   time_delta, n_iterations))
    print(forecast_result)
    forecast_result.to_csv('reports/forecast_result.csv', index=True)


if __name__ == '__main__':
    main()