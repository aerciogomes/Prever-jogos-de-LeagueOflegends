#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 21:22:49 2018

@author: alexandre costa
"""
"""A da vitória seria o y e ele tentaria classificar em vitória e derrota. 
Seria um classificador. Pro método, tenta com random forest, aumentando o número preditores. 
Tenta com 10,100,200. Tenta tb svm, kernel rbf. Pra ver no que dá"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from keras.layers import Dropout



df = pd.read_csv('LeagueofLegends.csv')
bans = pd.read_csv('bans.csv')
gold = pd.read_csv('gold.csv')
kills = pd.read_csv('kills.csv')
monsters = pd.read_csv('monsters.csv')
structures = pd.read_csv('structures.csv')



#### Trabalhando apenas com 2017 e 2018 pra exportar e tratar no excel hehehe
datateste = df[df['Year'] > 2016]
datateste.to_csv('goldvitoria.csv')

####### importando base tratada
vitoriaouro = pd.read_csv('pergunta1.csv')

Xtudo = vitoriaouro.iloc[:, 19:20]
ytudo = vitoriaouro.iloc[:, -1]
ytudo.value_counts().plot.bar()

######################BASE DO GOLD POSITVO
datatestepositivo = vitoriaouro[vitoriaouro['minute20'] >= 0]

Xpositivo = datatestepositivo.iloc[:, 19:20]
ypositivo = datatestepositivo.iloc[:, 20]

ypositivo.value_counts().plot.bar()

bloxplot = Xpositivo.boxplot()

##balaceando a base positiva
tamanho_amostra = 886
# obtendo as classes da base de dados
classes = datatestepositivo['result'].unique()
# calculando a quantidade de amostras por classe neste exemplo, serao amostradas as mesmas quantidades para cada classe
qtde_por_classe = round(tamanho_amostra / len(classes))
# nesta lista armazenaremos, para cada classe, um pandas.DataFrame com suas amostras
amostras_por_classe = []  
for c in classes:
# obtendo os indices do DataFrame
# cujas instancias pertencem a classe c
    indices_c = datatestepositivo['result'] == c
# extraindo do DataFrame original as observacoes da
# classe c (obs_c sera um DataFrame tambem)
    obs_c = datatestepositivo[indices_c]
# extraindo a amostra da classe c
# caso deseje-se realizar amostragem com reposicao
# ou caso len(obs_c) < qtde_por_classe, pode-se
# informar o parametro replace=True
    amostra_c = obs_c.sample(qtde_por_classe)
# armazenando a amostra_c na lista de amostras
    amostras_por_classe.append(amostra_c)
# concatenando as amostras de cada classe em
# um único DataFrame
datatestepositivo = pd.concat(amostras_por_classe)

from sklearn.utils import shuffle
datatestepositivo = shuffle(datatestepositivo)
########################################################

##########BASE DO GOLD POSITIVO E SÓ VITORIAS(BLLUE)
positivowin = datatestepositivo[datatestepositivo['result'] > 0]
Xpositivowin = positivowin.iloc[:, 2:20]
ypositivowin = positivowin.iloc[:, 20]

boxplot = Xpositivowin.boxplot()
plt.xlabel('Vantagem de ouro do blue na vitoria')
##################################### 

##########BASE DO GOLD NEGATIVO E SÓ DERROTA(RED)
negativolose = datatestenegativo[datatestenegativo['result'] > 0]
Xnegativolose = negativolose.iloc[:, 19:20]
ynegativolose = negativolose.iloc[:, 20]

boxplot = Xnegativolose.boxplot()
plt.xlabel('Vantagem de ouro do red na derrota')
##################################### 

##########BASE DO GOLD POSITIVO E SÓ DERROTAS(BLUE)
positivolose = datatestepositivo[datatestepositivo['result'] <= 0]
Xpositivolose = positivolose.iloc[:, 19:20]
ypositivolose = positivolose.iloc[:, 20]

boxplot = Xpositivolose.boxplot()
plt.xlabel('Vantagem de ouro do blue na derrota')    
##################################### 

##########BASE DO GOLD NEGATIVO E SÓ VITÓRIAS(RED)
negativowin = datatestenegativo[datatestenegativo['result'] < 0]
Xnegativowin = negativowin.iloc[:, 19:20]
ynegativowin = negativowin.iloc[:, 20]

boxplot = Xnegativowin.boxplot()
plt.xlabel('Vantagem de ouro do red na vitoria')    
##################################### 

######################BASE DO GOLD NEGATIVO
datatestenegativo = vitoriaouro[vitoriaouro['minute20'] < 0]

Xnegativo = datatestenegativo.iloc[:, 19:20]
ynegativo = datatestenegativo.iloc[:, 20]
ynegativo.value_counts().plot.bar()

bloxplot = Xnegativo.boxplot()


##balaceando a base positiva
tamanho_amostra = 860
# obtendo as classes da base de dados
classes = datatestenegativo['result'].unique()
# calculando a quantidade de amostras por classe neste exemplo, serao amostradas as mesmas quantidades para cada classe
qtde_por_classe = round(tamanho_amostra / len(classes))
# nesta lista armazenaremos, para cada classe, um pandas.DataFrame com suas amostras
amostras_por_classe = []  
for c in classes:
# obtendo os indices do DataFrame
# cujas instancias pertencem a classe c
    indices_c = datatestenegativo['result'] == c
# extraindo do DataFrame original as observacoes da
# classe c (obs_c sera um DataFrame tambem)
    obs_c = datatestenegativo[indices_c]
# extraindo a amostra da classe c
# caso deseje-se realizar amostragem com reposicao
# ou caso len(obs_c) < qtde_por_classe, pode-se
# informar o parametro replace=True
    amostra_c = obs_c.sample(qtde_por_classe)
# armazenando a amostra_c na lista de amostras
    amostras_por_classe.append(amostra_c)
# concatenando as amostras de cada classe em
# um único DataFrame
datatestenegativo = pd.concat(amostras_por_classe)

from sklearn.utils import shuffle
datatestenegativo = shuffle(datatestenegativo)
########################################################




####### importando base tratada
vitoriaouro = pd.read_csv('pergunta1.csv')


X = vitoriaouro.iloc[:, -1]
y = vitoriaouro.iloc[:, 20]

####verificando se tinha outro valor além de 0
X['minute1'].value_counts().plot.bar()

##dropando coluna minute1
X.drop(['minute1'],axis=1, inplace=True)

###verificando por nan
null_columns=vitoriaouro.columns[vitoriaouro.isnull().any()]
vitoriaouro.drop(vitoriaouro[vitoriaouro.isnull().any(axis=1)][null_columns].head())


##########################
#TESTANDO REDE NEURAL
##SPLITANDO OS DADOS TREINO TEST E VAL
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 0)

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)
y_val = keras.utils.to_categorical(y_val)


classifier = Sequential()


classifier.add(Dense(units = 100, kernel_initializer = 'uniform', activation = 'relu', input_dim = 19))
classifier.add(Dense(units= 100, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units= 100, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units= 2, kernel_initializer = 'uniform', activation = 'softmax'))


classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


classifier.fit(X_train, y_train, nb_epoch = 100,  validation_data=(X_val, y_val))

history = classifier.fit(X_train, y_train, nb_epoch = 100,  validation_data=(X_val, y_val)) ##pra poder plotar o grafico
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
######################



########KNN
# dividindo o dataset em test e treino
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Xnegativo, ynegativo, test_size = 0.30)

#Feature Scaling
from sklearn.preprocessing import RobustScaler
sc = RobustScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=11)

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
################################

###############################
###ARVORE DECISAO
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Xpositivo, ypositivo, test_size = 0.30)


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
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Xnegativo, ynegativo, test_size = 0.30)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy')
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









        
