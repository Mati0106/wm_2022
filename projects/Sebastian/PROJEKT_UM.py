import pandas as pd
from sklearn import datasets
from pandas_profiling import ProfileReport

data = pd.read_csv(r'C:/Users/sebma/Documents/GitHub/Sebastian/Data/water_potability.csv')
df = pd.DataFrame(data=data.data, columns=data.feature_names)
#profile = ProfileReport(df, title="Pandas Profiling Report", explorative=True)
#profile.to_file("your_report.pdf")

# RandomSearch, GridSearch, Optuna

















