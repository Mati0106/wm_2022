"""
Title: Kaggle scraper class
Author: Jedrzej Smulski
Date created: 2022-04-05
Last modified: 2022-04-26
Description: Class for scraping data from kaggle
"""


import json
import os
from typing import List
import shutil
import pandas as pd
import logging
from datetime import datetime

with open(os.environ['KAGGLEAUTHPATH']) as f:
   kaggle_auth = json.load(f)

os.environ['KAGGLE_USERNAME'] = kaggle_auth['username']
os.environ['KAGGLE_KEY'] = kaggle_auth['key']

logging.basicConfig(level=20)

# slabo ze ten import musi byc tutaj...
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

#api.dataset_list_files('arashnic/hr-analytics-job-change-of-data-scientists').files
# import pandas as pd
# print(type(pd.read_csv))

class KaggleScraper:

   def __init__(self, dataset: str):
      self.dataset = dataset

   def load_to_df(self, file_name: str, fun_to_load = pd.read_csv, verbose: bool=True)->pd.DataFrame:
      """
      :param file_name:
      :param fun_to_load:
      :param verbose:
      :return:
      :rtype: pd.DataFrame
      """
      if verbose:
         logging.info('Loading data to DateFrame...')
         tic = datetime.now()
      self.load_to_file(file_name, location=".", verbose=False)


      try:
         df = fun_to_load(file_name)
         os.remove(file_name)
      except:
         shutil.unpack_archive(file_name+'.zip', '.')
         os.remove(file_name+'.zip')
         df = fun_to_load(file_name)
         os.remove(file_name)
      if verbose:
         logging.info(f'DataFrame loaded in {datetime.now() - tic}')
      return df

   def load_to_file(self, file_name: str, location: str="data", verbose: bool=True)->None:
      """

      :param file_name:
      :param location:
      :param verbose:
      :return:
      """
      if verbose:
         logging.info('Loading data...')
      tic = datetime.now()
      api.dataset_download_file(self.dataset, file_name=file_name, path=f"{location}")
      if verbose:
         logging.info("Data downloaded.")
         logging.info(f"Data downloaded in {datetime.now()-tic}")

   def show_files(self)->List[str]:
      """

      :return:
      """
      return api.dataset_list_files(self.dataset).files

# sc = KaggleScraper("arashnic/hr-analytics-job-change-of-data-scientists")
# #sc.load_to_file(file_name="aug_test.csv")
# df = sc.load_to_df(file_name="aug_test.csv", fun_to_load=pd.read_csv)
# print(df.columns)
# print(df.shape)