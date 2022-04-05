import json
import os
import pandas as pd

with open('sjedrek/.kaggle/kaggle.json') as f:
   kaggle_auth = json.load(f)

os.environ['KAGGLE_USERNAME'] = kaggle_auth['username']
os.environ['KAGGLE_KEY'] = kaggle_auth['key']
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

api.dataset_list_files('arashnic/hr-analytics-job-change-of-data-scientists').files

def srap_data(dataset: str, file_name: str) -> pd.DataFrame:
   """

   :param dataset:
   :param file_name:
   :return:
   """

   return pd.DataFrame()

print(srap_data('', ''))