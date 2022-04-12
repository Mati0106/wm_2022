import pandas as pd
from sklearn import datasets
from pandas_profiling import ProfileReport
import ipywidgets as widgets
from ipywidgets import interact
import plotly.express as px
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import svm
import numpy as np

#----------------------------------
# Wczytanie danych
data = pd.read_csv(r'C:/Users/sebma/Documents/GitHub/Sebastian/Data/water_potability.csv')
df = pd.DataFrame(data=data)
profile = ProfileReport(df, title="Woda_raport", explorative=True)
profile.to_file("Woda_raport.pdf")


#--------------------------------
# Podstawowe informacje
df.info()
# mamy braki danych
print("Ilość duplikatów: ", df.duplicated().sum())
# nie mamy duplikatów

# jaki jest procentowy udzial pustych komorek
puste = df.isnull().sum().sort_values(ascending=False)
puste_procent = (df.isnull().sum() / df.isnull().count() * 100).sort_values(ascending=False)
braki = pd.concat([puste, puste_procent], axis=1, keys=['Łącznie', 'Procent'])
print(braki)

# poki co je usuwamy
df.dropna(inplace=True)

# ==============================Wykresy============================
# @interact(kolumna=['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity',
#                    'Organic_carbon', 'Trihalomethanes', 'Turbidity', 'Potability'])
# def wykresy(kolumna):
#     fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(15, 5))
#     axs[0].boxplot(df[kolumna])
#     axs[0].set_xlabel("Boxplot zmiennej " + str(kolumna))
#     axs[1].hist(df[kolumna])
#     axs[1].set_xlabel("Histogram zmiennej " + str(kolumna))
#     fig.suptitle("Zmienna " + str(kolumna).upper(), fontsize=16)
#     return plt.show()


#  Statystyki
for col in df.columns:
    print(df[col].describe().apply(lambda x: format(x, 'f')).T, '\n','======================')

df.describe().round(2).T


# metodą IQR zamieniamy anomalie na wasy
for col in df.columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    low = Q1 - 1.5 * IQR
    up = Q3 + 1.5 * IQR
    print("Dla zmiennej ", col, "low =" ,low, " , up =", up)
    lowp = df[df[col]<low].shape[0]/df.shape[0]*100
    upp = df[df[col]>up].shape[0]/df.shape[0]*100
    print("Dla zmiennej ", col, "wartosci odstajacych z gory jest: ", upp,"%.")
    print("Wartosci odstajacych z dolu jest: ", lowp, "%",'\n')

for col in df.columns[:-1]:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    low = Q1 - 1.5 * IQR
    up = Q3 + 1.5 * IQR
    df.loc[df[col] > up, col] = up
    df.loc[df[col] < low, col] = low


# Korelacja
korelacja_P = df.corr('pearson')
print(korelacja_P)

korelacja_S = df.corr('spearman')
print(korelacja_S)

#tworzymy macierz trójkątną i wyświetlamy wspóczynnik korelacji większy od 0.5
korelacja_P_tr = korelacja_P.where(np.triu(np.ones(korelacja_P.shape, dtype=np.bool), k=1)).stack().sort_values()
korelacja_P_tr[abs(korelacja_P_tr)>0.1]


# plt.figure(figsize=(15,15))
# ax = sns.heatmap(korelacja_P, square=True, annot=True, fmt='.2f')
# ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
# plt.show()

# plt.figure(figsize=(15,15))
# ax = sns.heatmap(korelacja_S, square=True, annot=True, fmt='.2f')
# ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
# plt.show()


# Podział zbioru

X = df.drop(labels=['Potability'], axis=1)
Y = df.iloc[:, 9]

clf = svm.SVC()

clf.fit(X, Y)

clf.predict([[7., 196., 21868., 7., 333., 426., 14., 66., 4.]])

X_ucz, X_test, Y_ucz, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2, random_state=12345)
print(X_ucz.shape)
print(X_test.shape)
print(Y_ucz.shape)
print(Y_test.shape)



# zbalansowac dane, uzyc metryk sprawdzajacych model, lepiej ogarnac nulle i wartosci odstajace