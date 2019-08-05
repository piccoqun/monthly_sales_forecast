import pandas as pd
import os
import pickle

#pd.set_option('display.max_columns', 10)
#pd.set_option('display.max_rows',100)


def load_data():

    xy_train_df_path = 'data/xy_train_df.pkl'
    forecast_dict_path = 'data/forecast_dict.pkl'

    if os.path.exists(xy_train_df_path) & os.path.exists(forecast_dict_path):
        with open(xy_train_df_path, 'rb') as handle_df:
            xy_train_df = pickle.load(handle_df)
        with open(forecast_dict_path, 'rb') as handle_dict:
            forecast_dict = pickle.load(handle_dict)
    else:
        # read data from excels
        act_lo_df = pd.read_excel('data/Exercise - ACT and LO Monthly - FOR CANDIDATE-SENT - SHORT.xlsx')  # (72,4)
        # attention: the first sales sheet is an empty 'hiddensheet'
        sales_df = pd.read_excel('data/Exercise - Daily Sales - FOR CANDIDATE-SENT - SHORT.xlsx', sheet_name=None)[
            'Daily Sales']  # (634, 4)
        days_dict = pd.read_excel('data/Exercise - Working Days calendar - FOR CANDIDATE-SENT - SHORT.xlsx',
                                  sheet_name=None)

        # cleaning: usually includes dropna, dropduplicates, redefine data types, reindex
        weekdays_df_raw = days_dict['Weekdays'].copy()  # (42,11)
        weekdays_df = df_reform(weekdays_df_raw)  # (38, 10)
        weekdays_df.rename(columns={'Country 1': 'Working Days'}, inplace=True)
        weekdays_df['Period'] = pd.to_datetime(weekdays_df['Month-Year']).dt.to_period('M')
        weekdays_df['Qtr'] = weekdays_df['Period'].dt.quarter
        weekdays_quarter_df = pd.get_dummies(weekdays_df['Qtr'], prefix='Q')
        weekdays_df = pd.concat([weekdays_df, weekdays_quarter_df], axis=1)
        weekdays_df.set_index('Period', inplace=True)

        calendar_df_raw = days_dict['Calendar'].copy()  # (1159, 9)
        calendar_df = df_reform(calendar_df_raw)  # (1155, 9)
        calendar_df.columns = ['Month Year', 'Date', 'Date2', 'Weekday', 'Year', 'Month', 'Month_cap', 'Day', 'Holiday']
        calendar_df.index = pd.to_datetime(calendar_df['Date'])
        calendar_df['Period'] = calendar_df.index.to_period('M')
        calendar_df = calendar_df[['Weekday', 'Holiday', 'Period']]  # (1155, 3)
        holidays_df = calendar_df.groupby('Period')['Holiday'].value_counts().unstack()

        # reform data into monthly time series for the model
        act_lo_df['Period'] = pd.to_datetime(act_lo_df[['Year', 'Month']].assign(Day=1)).dt.to_period('M')
        act_lo_df.set_index('Period', inplace=True)
        act_df = act_lo_df[act_lo_df['Submission'] == 'Actual'].rename(columns={'Country 1 - Brand A': 'Actuals'})
        lo_df = act_lo_df[act_lo_df['Submission'].str.contains('LO')].rename(columns={'Country 1 - Brand A': 'LO'})

        xy_df = pd.concat([act_df['Actuals'], lo_df['LO'], weekdays_df[['Working Days','Q_1','Q_2','Q_3','Q_4']],
                           holidays_df[0]], axis=1)  # (38, 4)
        xy_df.rename(columns={0: 'Holidays'}, inplace=True)

        sales_df.index = pd.to_datetime(sales_df['Posting Date'], format='%d.%m.%Y')
        sales_df = pd.concat([sales_df, calendar_df], axis=1).sort_index()  # sort in ascending dates
        monthly_sales_df = monthly_average(sales_df)  # (38, 7)

        weekdays_ls = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        xy_df = pd.concat([xy_df, monthly_sales_df[weekdays_ls]], axis=1)  # (38, 9)
        xy_train_df = xy_df.iloc[:-4].sort_index().astype('float64')  # (34, 9)

        # produce forecasting training set at each reporting date
        sales_forecast_df = sales_df.loc['2017-11-01':'2018-03-01']  # (120, 7)
        sales_forecast_date_df = sales_forecast_df.loc[sales_forecast_df.Holiday == 1]
        sales_forecast_date_index = sales_forecast_date_df.loc['2017-11-15':'2017-11-25'].index \
            .union(sales_forecast_date_df.loc['2017-12-15':'2017-12-25'].index) \
            .union(sales_forecast_date_df.loc['2018-01-15':'2018-01-25'].index) \
            .union(sales_forecast_date_df.loc['2018-02-15':'2018-02-25'].index)  # 28
        forecast_dict = {}
        columns_excl_weekdays_ls = list(set(xy_df.columns) - set(weekdays_ls))

        for date in sales_forecast_date_index:
            append_train_df = monthly_average(sales_forecast_df.loc[:date])[weekdays_ls]
            append_index = append_train_df.index
            xy_train_new_df = xy_train_df.append(append_train_df, sort=False)
            # in pandas version >0.23.0 there will be warning if there is no sort indicated
            # sort = True makes alphanumerically columns when two dataframe's columns are not aligned
            xy_train_new_df.loc[append_index, columns_excl_weekdays_ls] = xy_df.loc[
                append_index, columns_excl_weekdays_ls].copy()
            forecast_dict[date] = xy_train_new_df.astype('float64')

        with open(xy_train_df_path, 'wb') as handle_df:
            pickle.dump(xy_train_df, handle_df, protocol=pickle.HIGHEST_PROTOCOL)

        with open(forecast_dict_path, 'wb') as handle_dict:
            pickle.dump(forecast_dict, handle_dict, protocol=pickle.HIGHEST_PROTOCOL)

    return xy_train_df, forecast_dict

def df_reform(df):
    # this function is to delete empty rows and columns and remove column name rows
    # df has two rows of column names
    df_new = df.copy() # make copy so that it doesnt affect old data
    df_new.dropna(how='all', inplace=True)
    df_new.dropna(how='all', axis=1, inplace=True)
    df_new.rename(columns = df_new.iloc[1],inplace=True)
    df_new.drop(df_new.index[[0,1]],inplace=True)
    return df_new

def monthly_average(sales_df):
    # this function is to calculate monthly average sales from daily sales
    sales_df_groupby = sales_df.groupby(['Period', 'Weekday'])[['Daily Sales']].mean()
    monthly_sales_df = sales_df_groupby.unstack(level=-1)
    # drop multi level of columns for further simpler quotations
    monthly_sales_df.columns = monthly_sales_df.columns.droplevel(0)
    return monthly_sales_df

