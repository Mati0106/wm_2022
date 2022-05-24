"""
Title: Data preparation functions
Author: Jedrzej Smulski
Date created: 2022-04-26
Last modified: 2022-04-26
Description: Data preparation functions
"""

def df_first_look(df):
    print(50*'=')
    print(f'{df.head()}')
    print(50*'=')
    print(f'COLUMNS: \n {df.columns.values}')
    print(50 * '=')
    print(f'SHAPE: \n {df.shape}')
    print(50*'=')
    print(f'{df.info()}')
    print(50*'=')
    print(f'{df.describe()}')
    print(50*'=')
    print(f'{df.describe(include=object).T}')
    print(50*'=')
    print(f'UNIQUE VALUES IN COLUMNS')
    for c in df.select_dtypes(include=object).columns.values:
        print(c)
        print(df[c].unique())
        print(20*'-')


def nan_sim_replace():
    pass

def nan_delete(df):
    return df.dropna(inplace=True)

# replace by specific value or function like median/mean
def nan_val_replace(df, column, method):

    # replace by specific value
    if method==1:
        df.fillna(inplace=True)