"""
Title: HR Analytics: Job Change of Data Scientists
Author: Jedrzej Smulski
Date created: 2022-04-26
Last modified: 2022-04-26
Description: Data analysis, visualisation, feature engineering and model training for job change classification.
"""
"""
## Introdution


"""
"""
## Setup
"""
import os

import pandas as pd
from sjedrek.src.data_preparation import df_first_look, nan_delete
from sjedrek.src.kaggle_sraper import KaggleScraper


FILENAME='aug_train.csv'
FUN_TO_LOAD=pd.read_csv
VERBOSE=True

"""
## Prepare the data

Let's get a short view of data
"""
sc = KaggleScraper(os.environ['KAGGLEPROJECT'])
df_raw = sc.load_to_df(file_name=FILENAME, fun_to_load=FUN_TO_LOAD, verbose=VERBOSE)
df_first_look(df_raw)

