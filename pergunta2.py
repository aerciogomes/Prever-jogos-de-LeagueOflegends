#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 01:10:44 2018

@author: alexandre
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('testesemponto.csv')

dfteste = df/1000 # so vai ser preciso essa divisao nos barons
"""FALTA ORDENAR POR INSTANCIAS"""

df = pd.read_csv('LeagueofLegends.csv')
datateste = df[df['Year'] > 2016]
datateste.to_csv('basetratada.csv')


testedragao = pd.read_csv('testedragao.csv')
testedragao = testedragao/1000 # so vai ser preciso essa divisao nos barons
dragaosorted = testedragao.sort_values(by=['rDragon', 'rDragon.1', 'rDragon.2', 'rDragon.3', 'rDragon.4', 'rDragon.5'])
##ele ordena o resto com base na primeira passada como parâmetro

####TRANSFORMA LINHA EM COLUNA PARA ORDENAR
dragon1 = testedragao.iloc[:, 0:2]    
dragonT = dragon1.T
dragonT = dragonT.apply(lambda x: x.sort_values(na_position='first').values)
dragonsorted = dragonT.T

#######BARAO BLUE
baronblue = df.iloc[:,0:5]
baronblueT = baronblue.T
baronblueT = baronblueT.apply(lambda x: x.sort_values(na_position='first', ascending=[False]).values)
baronsorted = baronblueT.T
baronsorted = baronsorted/1000

bresult = pd.read_csv('bresult.csv')
goldbaron = pd.read_csv('goldbaron.csv')
goldbaron = goldbaron.iloc[:,3:31]
baronsortedblue = pd.concat([baronsorted, goldbaron], axis=1)
baronsortedblue = pd.concat([baronsortedblue, bresult], axis=1)

baronsortedblue = baronsortedblue.iloc[:,4:34]
baronsortedblue = baronsortedblue.dropna()

##baronsortedblue = baronsortedblue.dropna(subset = ['bBaron.4'])
##baronsortedblue = baronsortedblue.dropna(subset = ['goldbaron'])
Xbaronblue = baronsortedblue.iloc[:, 0:29]
Ybaronblue = baronsortedblue.iloc[:, -1]

boxplot = Xbaronblue.boxplot()
Ybaronblue.value_counts().plot.bar() ###1754 | 325 -- 650

print( baronsorted[baronsorted['bBaron.4'] < 10])

###balanceando
tamanho_amostra = 514

classes = baronsortedblue['bResult'].unique()
# calculando a quantidade de amostras por classe neste exemplo, serao amostradas as mesmas quantidades para cada classe
qtde_por_classe = round(tamanho_amostra / len(classes))
# nesta lista armazenaremos, para cada classe, um pandas.DataFrame com suas amostras
amostras_por_classe = []  
for c in classes:
# obtendo os indices do DataFrame
# cujas instancias pertencem a classe c
    indices_c = baronsortedblue['bResult'] == c
# extraindo do DataFrame original as observacoes da
# classe c (obs_c sera um DataFrame tambem)
    obs_c = baronsortedblue[indices_c]
# extraindo a amostra da classe c
# caso deseje-se realizar amostragem com reposicao
# ou caso len(obs_c) < qtde_por_classe, pode-se
# informar o parametro replace=True
    amostra_c = obs_c.sample(qtde_por_classe)
# armazenando a amostra_c na lista de amostras
    amostras_por_classe.append(amostra_c)
# concatenando as amostras de cada classe em
# um único DataFrame
baronsortedblue = pd.concat(amostras_por_classe)

from sklearn.utils import shuffle
baronsortedblue = shuffle(baronsortedblue)



######BARAO RED
baronred = df.iloc[:,6:11]
baronredT = baronred.T
baronredT = baronredT.apply(lambda x: x.sort_values(na_position='first', ascending=[False]).values)
baronsortedred = baronredT.T
baronsortedred = baronsortedred/1000

bresult = pd.read_csv('bresult.csv')
goldbaronred = pd.read_csv('goldbaron.csv')
baronsortedreddd = pd.concat([baronsortedred, gold], axis=1)
baronsortedreddd = pd.concat([baronsortedreddd, bresult], axis=1)

baronsortedred = baronsortedreddd.dropna(subset = ['rBaron.4'])
baronsortedred = baronsortedred.dropna(subset = ['gold'])
Xbaronred = baronsortedred.iloc[:, 4:6]
Ybaronred = baronsortedred.iloc[:, -1]

Ybaronred.value_counts().plot.bar()

###balanceando
tamanho_amostra = 734

classes = baronsortedred['bResult'].unique()
# calculando a quantidade de amostras por classe neste exemplo, serao amostradas as mesmas quantidades para cada classe
qtde_por_classe = round(tamanho_amostra / len(classes))
# nesta lista armazenaremos, para cada classe, um pandas.DataFrame com suas amostras
amostras_por_classe = []  
for c in classes:
# obtendo os indices do DataFrame
# cujas instancias pertencem a classe c
    indices_c = baronsortedred['bResult'] == c
# extraindo do DataFrame original as observacoes da
# classe c (obs_c sera um DataFrame tambem)
    obs_c = baronsortedred[indices_c]
# extraindo a amostra da classe c
# caso deseje-se realizar amostragem com reposicao
# ou caso len(obs_c) < qtde_por_classe, pode-se
# informar o parametro replace=True
    amostra_c = obs_c.sample(qtde_por_classe)
# armazenando a amostra_c na lista de amostras
    amostras_por_classe.append(amostra_c)
# concatenando as amostras de cada classe em
# um único DataFrame
baronsortedred = pd.concat(amostras_por_classe)

from sklearn.utils import shuffle
baronsortedred = shuffle(baronsortedred)

##########    KNN

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Xtodoredteste, ytodoredteste, test_size = 0.30)

from sklearn.preprocessing import RobustScaler
sc = RobustScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=25)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#acuracia
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred, normalize=True)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = knn, X = X_train, y = y_train, cv = 10)
accuracies.mean()

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'n_neighbors': [1, 3, 5, 7, 11, 13, 15, 17, 19, 21,23,25,27,29,31,33,35,37]}]
grid_search = GridSearchCV(estimator = knn,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

###ARVORE DECISAO
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Xtodoredteste, ytodoredteste, test_size = 0.30)


from sklearn.tree import DecisionTreeClassifier
#from sklearn import tree
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)

y_pred = clf_gini.predict(X_test)
acc = accuracy_score(y_test, y_pred, normalize=True)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = clf_gini, X = X_train, y = y_train, cv = 10)
accuracies.mean()

####################################

###################
##########RANDON FOREST
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Xtodoredteste, ytodoredteste, test_size = 0.30)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 200, criterion = 'entropy')
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#acuracia
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred, normalize=True)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()

from sklearn.metrics import f1_score
f1_score(y_test, y_pred, average='micro')  
specificity1 = cm[1,1]/(cm[1,0]+cm[1,1])
sensitivity1 = cm[0,0]/(cm[0,0]+cm[0,1])

################################################################################################################################################

########DRAGONS
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

testedragao = pd.read_csv('dragons.csv')

#####blue dragon
dragonblue = testedragao.iloc[:,0:8]
dragonT = dragonblue.T
dragonT = dragonT.apply(lambda x: x.sort_values(na_position='first', ascending=[False]).values)
dragonsorted = dragonT.T

accuries = dragonsorted.iloc[:,-1]
accuries.mean()

bresult = pd.read_csv('bresult.csv')
golddragon = pd.read_csv('golddragon.csv')

dragonsorted = dragonsorted.iloc[:,-1]
golddragon = golddragon.iloc[:,3:19]

dragononsortedblue = pd.concat([dragonsorted, golddragon], axis=1)
dragononsortedblue = pd.concat([dragononsortedblue, bresult], axis=1)

dragononsortedblue = dragononsortedblue.dropna(subset = ['bdragon8'])
dragononsortedblue = dragononsortedblue.dropna(subset = ['gold'])

Xdragonblue = dragononsortedblue.iloc[:, 7:9]
Ydragonblue = dragononsortedblue.iloc[:, -1]

Ydragonblue.value_counts().plot.bar()

###balanceando
tamanho_amostra = 1790

classes = dragononsortedblue['bResult'].unique()
# calculando a quantidade de amostras por classe neste exemplo, serao amostradas as mesmas quantidades para cada classe
qtde_por_classe = round(tamanho_amostra / len(classes))
# nesta lista armazenaremos, para cada classe, um pandas.DataFrame com suas amostras
amostras_por_classe = []  
for c in classes:
# obtendo os indices do DataFrame
# cujas instancias pertencem a classe c
    indices_c = dragononsortedblue['bResult'] == c
# extraindo do DataFrame original as observacoes da
# classe c (obs_c sera um DataFrame tambem)
    obs_c = dragononsortedblue[indices_c]
# extraindo a amostra da classe c
# caso deseje-se realizar amostragem com reposicao
# ou caso len(obs_c) < qtde_por_classe, pode-se
# informar o parametro replace=True
    amostra_c = obs_c.sample(qtde_por_classe)
# armazenando a amostra_c na lista de amostras
    amostras_por_classe.append(amostra_c)
# concatenando as amostras de cada classe em
# um único DataFrame
dragononsortedblue = pd.concat(amostras_por_classe)

from sklearn.utils import shuffle
dragononsortedblue = shuffle(dragononsortedblue)

#####Red Dragon
dragonred = testedragao.iloc[:,8:15]
dragonTred = dragonred.T
dragonTred = dragonTred.apply(lambda x: x.sort_values(na_position='first', ascending=[False]).values)
dragonsortedred = dragonTred.T

accuries = dragonsortedred.iloc[:,-1]
accuries.mean()

bresult = pd.read_csv('bresult.csv')
goldred = pd.read_csv('golddragonred.csv')

dragonsortedred = pd.concat([dragonsortedred, goldred], axis=1)
dragonsortedred = pd.concat([dragonsortedred, bresult], axis=1)

dragonsortedred = dragonsortedred.dropna(subset = ['rDragon7'])
dragonsortedred = dragonsortedred.dropna(subset = ['gold'])

Xdragonred = dragonsortedred.iloc[:, 6:8]
Ydragonred = dragonsortedred.iloc[:, -1]

Ydragonred.value_counts().plot.bar()

###balanceando
tamanho_amostra = 2462

classes = dragonsortedred['bResult'].unique()
# calculando a quantidade de amostras por classe neste exemplo, serao amostradas as mesmas quantidades para cada classe
qtde_por_classe = round(tamanho_amostra / len(classes))
# nesta lista armazenaremos, para cada classe, um pandas.DataFrame com suas amostras
amostras_por_classe = []  
for c in classes:
# obtendo os indices do DataFrame
# cujas instancias pertencem a classe c
    indices_c = dragonsortedred['bResult'] == c
# extraindo do DataFrame original as observacoes da
# classe c (obs_c sera um DataFrame tambem)
    obs_c = dragonsortedred[indices_c]
# extraindo a amostra da classe c
# caso deseje-se realizar amostragem com reposicao
# ou caso len(obs_c) < qtde_por_classe, pode-se
# informar o parametro replace=True
    amostra_c = obs_c.sample(qtde_por_classe)
# armazenando a amostra_c na lista de amostras
    amostras_por_classe.append(amostra_c)
# concatenando as amostras de cada classe em
# um único DataFrame
dragonsortedred = pd.concat(amostras_por_classe)

from sklearn.utils import shuffle
dragonsortedred = shuffle(dragonsortedred)

################################################################################################################################################


#######JUNTANDO TUDO BLUE

baronsortedblue = baronsorted.iloc[:,-1]
baronsortedblue = pd.concat([baronsortedblue, goldbaron], axis=1)

dragonsortedblue = dragonsorted.iloc[:,-1]
dragonsortedblue = pd.concat([dragonsortedblue, golddragon], axis=1)

tudoblue = baronsortedblue
tudoblueteste = pd.concat([tudoblue, dragonsortedblue], axis=1)
tudoblueteste = pd.concat([tudoblueteste, bresult], axis=1)
tudoblueteste = tudoblueteste.dropna(subset = ['bBaron.4'])
tudoblueteste = tudoblueteste.dropna(subset = ['goldbaron'])
tudoblueteste = tudoblueteste.dropna(subset = ['golddragon'])
tudoblueteste = tudoblueteste.dropna(subset = ['bdragon8'])

Xtudoblue = tudoblueteste.iloc[:, 0:4]
Ytudoblue = tudoblueteste.iloc[:, -1]

Ytudoblue.value_counts().plot.bar()

#############################################################################
bresult = bresult.dropna(subset = ['bResult'])
bresult = bresult.iloc[:,-1]
bresult.value_counts().plot.bar()
#############################################################################

###balanceando
tamanho_amostra = 416

classes = tudoblueteste['bResult'].unique()
# calculando a quantidade de amostras por classe neste exemplo, serao amostradas as mesmas quantidades para cada classe
qtde_por_classe = round(tamanho_amostra / len(classes))
# nesta lista armazenaremos, para cada classe, um pandas.DataFrame com suas amostras
amostras_por_classe = []  
for c in classes:
# obtendo os indices do DataFrame
# cujas instancias pertencem a classe c
    indices_c = tudoblueteste['bResult'] == c
# extraindo do DataFrame original as observacoes da
# classe c (obs_c sera um DataFrame tambem)
    obs_c = tudoblueteste[indices_c]
# extraindo a amostra da classe c
# caso deseje-se realizar amostragem com reposicao
# ou caso len(obs_c) < qtde_por_classe, pode-se
# informar o parametro replace=True
    amostra_c = obs_c.sample(qtde_por_classe)
# armazenando a amostra_c na lista de amostras
    amostras_por_classe.append(amostra_c)
# concatenando as amostras de cada classe em
# um único DataFrame
tudoblueteste = pd.concat(amostras_por_classe)

from sklearn.utils import shuffle
tudoblueteste = shuffle(tudoblueteste)
####balanceado off
FILTRO1=tudoblueteste['bResult'] == 1
data1 = tudoblueteste[FILTRO1]
FILTRO2=tudoblueteste['bResult'] == 0 
data2 = tudoblueteste[FILTRO2] 

from sklearn.utils import shuffle 
data1 = shuffle(data1) 
data1 = data1.sample(n = 500) 
data = data1.append(data2)

Xdata = data.iloc[:,0:4]
ydata = data.iloc[:,-1]

from imblearn.over_sampling import RandomOverSampler 
ros = RandomOverSampler(ratio='all')
X_res, y_res = ros.fit_sample(Xdata, ydata)

from sklearn.utils import shuffle
X_res = shuffle(X_res)
y_res = shuffle(y_res)

#############################################################################
#######JUNTANDO TUDO RED

baronsortedred = baronsortedred.iloc[:,-1]
baronsortedred = pd.concat([baronsortedred, goldbaronred], axis=1)

dragonsortedred = dragonsortedred.iloc[:,-1]
dragonsortedred = pd.concat([dragonsortedred, goldred], axis=1)

todored = baronsortedred
todoredteste = pd.concat([todored, dragonsortedred], axis=1)
todoredteste = pd.concat([todoredteste, bresult], axis=1)

todoredteste = todoredteste.dropna(subset = ['rBaron.4'])
todoredteste = todoredteste.dropna(subset = ['goldbaron'])
todoredteste = todoredteste.dropna(subset = ['gold'])
todoredteste = todoredteste.dropna(subset = ['rDragon7'])

"""todorednafrente = todoredteste[todoredteste['gold'] < 0]
todorednafrente = todoredteste[todoredteste['goldbaron'] < 0]
Xtteste = todorednafrente.iloc[:, 0:4]
Ytteste = todorednafrente.iloc[:, -1]
Ytteste.value_counts().plot.bar()"""

Xtudored = todoredteste.iloc[:, 0:4]
Ytudored = todoredteste.iloc[:, -1]

boxplot = Xtudored.boxplot()
Ytudored.value_counts().plot.bar()

###balanceando
tamanho_amostra = 624

classes = todoredteste['bResult'].unique()
# calculando a quantidade de amostras por classe neste exemplo, serao amostradas as mesmas quantidades para cada classe
qtde_por_classe = round(tamanho_amostra / len(classes))
# nesta lista armazenaremos, para cada classe, um pandas.DataFrame com suas amostras
amostras_por_classe = []  
for c in classes:
# obtendo os indices do DataFrame
# cujas instancias pertencem a classe c
    indices_c = todoredteste['bResult'] == c
# extraindo do DataFrame original as observacoes da
# classe c (obs_c sera um DataFrame tambem)
    obs_c = todoredteste[indices_c]
# extraindo a amostra da classe c
# caso deseje-se realizar amostragem com reposicao
# ou caso len(obs_c) < qtde_por_classe, pode-se
# informar o parametro replace=True
    amostra_c = obs_c.sample(qtde_por_classe)
# armazenando a amostra_c na lista de amostras
    amostras_por_classe.append(amostra_c)
# concatenando as amostras de cada classe em
# um único DataFrame
todoredteste = pd.concat(amostras_por_classe)

#############################################################################
#############################################################################
#############################################################################

df = pd.read_csv('testesemponto.csv')

baronblue = df.iloc[:,0:5]
baronblueT = baronblue.T
baronblueT = baronblueT.apply(lambda x: x.sort_values(na_position='first', ascending=[False]).values)
baronsorted = baronblueT.T
baronsorted = baronsorted/1000

bresult = pd.read_csv('bresult.csv')
goldbaron = pd.read_csv('goldbaron.csv')
goldbaron = goldbaron.iloc[:,3:31]
baronsortedblue = pd.concat([baronsorted, goldbaron], axis=1)

baronsortedblue = baronsortedblue.iloc[:,4:34]

testedragao = pd.read_csv('dragons.csv')

dragonblue = testedragao.iloc[:,0:8]
dragonT = dragonblue.T
dragonT = dragonT.apply(lambda x: x.sort_values(na_position='first', ascending=[False]).values)
dragonsorted = dragonT.T

bresult = pd.read_csv('bresult.csv')
golddragon = pd.read_csv('golddragon.csv')

dragonsorted = dragonsorted.iloc[:,-1]
golddragon = golddragon.iloc[:,3:19]

dragononsortedblue = pd.concat([dragonsorted, golddragon], axis=1)

todoblueteste = baronsortedblue
todoblueteste = pd.concat([todoblueteste,dragononsortedblue], axis=1)
todoblueteste = pd.concat([todoblueteste,bresult], axis=1)
todoblueteste = todoblueteste.dropna()

Xtodoblueteste = todoblueteste.iloc[:, 0:46]
ytodoblueteste = todoblueteste.iloc[:, -1]

boxplot = Xtodoblueteste.boxplot()

boxplot = Xtodoblueteste.boxplot()
ytodoblueteste.value_counts().plot.bar() ### 267

###balanceando
tamanho_amostra = 534

classes = todoblueteste['bResult'].unique()
# calculando a quantidade de amostras por classe neste exemplo, serao amostradas as mesmas quantidades para cada classe
qtde_por_classe = round(tamanho_amostra / len(classes))
# nesta lista armazenaremos, para cada classe, um pandas.DataFrame com suas amostras
amostras_por_classe = []  
for c in classes:
# obtendo os indices do DataFrame
# cujas instancias pertencem a classe c
    indices_c = todoblueteste['bResult'] == c
# extraindo do DataFrame original as observacoes da
# classe c (obs_c sera um DataFrame tambem)
    obs_c = todoblueteste[indices_c]
# extraindo a amostra da classe c
# caso deseje-se realizar amostragem com reposicao
# ou caso len(obs_c) < qtde_por_classe, pode-se
# informar o parametro replace=True
    amostra_c = obs_c.sample(qtde_por_classe)
# armazenando a amostra_c na lista de amostras
    amostras_por_classe.append(amostra_c)
# concatenando as amostras de cada classe em
# um único DataFrame
todoblueteste = pd.concat(amostras_por_classe)

from sklearn.utils import shuffle
todoblueteste = shuffle(todoblueteste)

FILTRO1=todoblueteste['bResult'] == 1
data1 = todoblueteste[FILTRO1]
FILTRO2=todoblueteste['bResult'] == 0 
data2 = todoblueteste[FILTRO2] 

from sklearn.utils import shuffle 
data1 = shuffle(data1) 
data1 = data1.sample(n = 500) 
data = data1.append(data2)

Xdata = data.iloc[:,0:46]
ydata = data.iloc[:,-1]

from imblearn.over_sampling import SMOTE, ADASYN
X_resampled, y_resampled = SMOTE(ratio='minority').fit_sample(Xdata, ydata)

from sklearn.utils import shuffle
X_resampled = shuffle(X_resampled)
y_resampled = shuffle(y_resampled)

np.count(y_resampled)

#############################################################################
#############################################################################
#############################################################################

df = pd.read_csv('testesemponto.csv')

baronred = df.iloc[:,6:11]
baronredT = baronred.T
baronredT = baronredT.apply(lambda x: x.sort_values(na_position='first', ascending=[False]).values)
baronsortedred = baronredT.T
baronsortedred = baronsortedred/1000

bresult = pd.read_csv('bresult.csv')
goldbaronred = pd.read_csv('goldbaron.csv')
goldbaronred = goldbaronred.iloc[:,3:31]
baronsortedreddd = pd.concat([baronsortedred, goldbaronred], axis=1)

baronsortedreddd = baronsortedreddd.iloc[:,4:33]

testedragao = pd.read_csv('dragons.csv')

dragonred = testedragao.iloc[:,8:15]
dragonTred = dragonred.T
dragonTred = dragonTred.apply(lambda x: x.sort_values(na_position='first', ascending=[False]).values)
dragonsortedred = dragonTred.T

accuries = dragonsortedred.iloc[:,-1]
accuries.mean()

bresult = pd.read_csv('bresult.csv')
goldred = pd.read_csv('golddragonred.csv')
goldred = goldred.iloc[:,3:18]

dragonsortedred = pd.concat([dragonsortedred, goldred], axis=1)
dragonsortedred = dragonsortedred.iloc[:,6:22]

todoredteste = baronsortedreddd
todoredteste = pd.concat([todoredteste,dragonsortedred], axis=1)
todoredteste = pd.concat([todoredteste,bresult], axis=1)
todoredteste = todoredteste.dropna()

Xtodoredteste = todoredteste.iloc[:, 0:45]
ytodoredteste = todoredteste.iloc[:, -1]

boxplot = todoredteste.boxplot()

boxplot = todoredteste.boxplot()
ytodoredteste.value_counts().plot.bar() ### 267

###balanceando
tamanho_amostra = 756

classes = todoredteste['bResult'].unique()
# calculando a quantidade de amostras por classe neste exemplo, serao amostradas as mesmas quantidades para cada classe
qtde_por_classe = round(tamanho_amostra / len(classes))
# nesta lista armazenaremos, para cada classe, um pandas.DataFrame com suas amostras
amostras_por_classe = []  
for c in classes:
# obtendo os indices do DataFrame
# cujas instancias pertencem a classe c
    indices_c = todoredteste['bResult'] == c
# extraindo do DataFrame original as observacoes da
# classe c (obs_c sera um DataFrame tambem)
    obs_c = todoredteste[indices_c]
# extraindo a amostra da classe c
# caso deseje-se realizar amostragem com reposicao
# ou caso len(obs_c) < qtde_por_classe, pode-se
# informar o parametro replace=True
    amostra_c = obs_c.sample(qtde_por_classe)
# armazenando a amostra_c na lista de amostras
    amostras_por_classe.append(amostra_c)
# concatenando as amostras de cada classe em
# um único DataFrame
todoredteste = pd.concat(amostras_por_classe)

from sklearn.utils import shuffle
todoredteste = shuffle(todoredteste)


#####PLOTANDO MATRIZ DE CONFUSÃO BONITINHA
import itertools
def plot_confusion_matrix(conf_matrix, classes,
                          normalize=False,
                          title='Matriz de Confusão',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    plt.imshow(conf_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = conf_matrix.max() / 2.
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(j, i, format(conf_matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

def compute_confusion_matrix(cnf_matrix, classes_):
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=classes_,
                      title='Matriz de Confusão')
    
classes_ = ['bResult = 1', 'bResult = 0']
compute_confusion_matrix(cm, classes_)





