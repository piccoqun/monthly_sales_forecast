# Monthly Sales Forecast

## Package
* python 3.6.3
* pandas 0.25.0
* numpy 1.17.0
* scikit-learn 0.21.3
* xgboost 0.90

## Code
Running main.py will generate result files in the format of png (pictures),
 csv (excels) and sav (models) in the folder 'reports'.

 'n_iterations' is set as 500 in the main.py, which may take very long
 time on CPU. One can tune this number based on his computer capacity.

Every run generates new reports covering old ones.

The feature extracted data is produced in the folder 'data' as pkl files.
If there are changes to original data, the two pickle files need to be
deleted so that the code generates new feature extracted data.