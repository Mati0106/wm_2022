from sklearn import datasets
from pandas_profiling import ProfileReport
import pandas as pd

data = datasets.load_iris() #pd.read_csv('sciezka_do_panstwa_danyhc.csv")
df = pd.DataFrame(data=data.data, columns=data.feature_names) #
profile = ProfileReport(df, title="Pandas Profiling Report", explorative=True)
profile.to_file("your_report.pdf")