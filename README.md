# Monthly Sales Forecast by Xgboost

Given daily sales and some monthly features, the task is to forecast
monthly sales 'Actuals' on specific weekdays (we call them the 'forecast dates').
In the business scenario,
as new data feature, such as daily sales, comes in everyday, the model is retrained and made
new forecast for each forecast date.

The main modelling process includes data preprocessing (reform data into
monthly time series, find weekdays and other basic cleaning), bootstrap,
xgboost training and randomized hyperparameter search.

The detailed analysis can be found in the pdf file 'Data Analysis Report'.

## Prediction Results
For each forecast date, model is retrained and the bootstrap prediction results
are saved in a folder named after the date. The prediction results are like
![picture](reports/model_15.01.2018/Predicted Actuals.png)

## Dependencies
* python 3.6.3
* pandas 0.25.0
* numpy 1.17.0
* scikit-learn 0.21.3
* xgboost 0.90

## Usage
Running main.py will generate result files in the format of png (pictures),
 csv (excels) and sav (models) in the folder 'reports'.

 'n_iterations' is set as 500 in the main.py, which may take very long
 time on CPU. One can tune this number based on his computer capacity.

Every run generates new reports covering old ones.

The feature extracted data is produced in the folder 'data' as pkl files.
If there are changes to original data, the two pickle files need to be
deleted so that the code generates new feature extracted data.