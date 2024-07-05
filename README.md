# Final-boston-project

[Summary](#Summary)
[Observations](#Observations)
[Slides](#Slides)
[Install](#Install)


## Summary
This project used a small dataframe about house price in Boston with several variables and try to develop an machine learning model that tries to predict this final price training the model in the other variables.



## Observations
* The existance of null values being 1/5 of the total rows makes it a problem in order to make initial conclusions or EDA
* Due to the limited sample range the studie couldn't be the extensive that it could be.
* Regarding the analysis of the variables I get the correlation between them and create histograms to see possible skews mistakes that we will later avoid by omiting some of the variables in the final model
* We get the weight of each of the variable in the final model which ends up being a RandomForest model with the higher R2 value (0.89)
* Finally we create an streamlit app that shows the trained model and makes able to create predictions


## Slides

The slides presenting the data and analysis can be found at: [slides]: ['https://docs.google.com/presentation/d/1q2Nl6OKYMD2ekxjmt2WCvNhJgCUPZvZaO2-S7Lr-MYM/edit#slide=id.g2ea19e8af24_0_401']

## Install

### Python

The data cleaning and machine learning in Python used the following packages:
* numpy
* pandas
* matplotlib
* seaborn
* scikit.learn
* joblib
* streamlit

The versions for the above packages can be found in the requirements text file.

## Development

Data was provided by Kaggle and was analysed, transformed and plotted in Jupyter Lab and streamlit.