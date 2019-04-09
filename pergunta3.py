#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 16:10:45 2018

@author: aercio
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('goldtop.csv')
df2 = pd.read_csv('goldmid.csv')
bresult = pd.read_csv('bresult.csv')

goldtop = df.iloc[:, 47:65]
goldmid = df2.iloc[:,47:65]

bloxplot = goldtop.boxplot()
bloxplot = goldmid.boxplot()

goldtop = goldtop[goldtop['goldtop20'] != 0]
goldmid = goldmid[goldmid['goldmid20'] != 0]

goldtopmid = pd.concat([goldtop, goldmid], axis=1)
goldtopmid = pd.concat([goldtopmid, bresult], axis=1)
goldtopmid = goldtopmid.dropna()
result = goldtopmid.iloc[:,-1]
result.value_counts().plot.bar()

Xtopmid = goldtopmid.iloc[:,0:36]
ytopmid = goldtopmid.iloc[:,-1]

###balanceando
tamanho_amostra = 3170

classes = goldtopmid['bResult'].unique()
# calculando a quantidade de amostras por classe neste exemplo, serao amostradas as mesmas quantidades para cada classe
qtde_por_classe = round(tamanho_amostra / len(classes))
# nesta lista armazenaremos, para cada classe, um pandas.DataFrame com suas amostras
amostras_por_classe = []  
for c in classes:
# obtendo os indices do DataFrame
# cujas instancias pertencem a classe c
    indices_c = goldtopmid['bResult'] == c
# extraindo do DataFrame original as observacoes da
# classe c (obs_c sera um DataFrame tambem)
    obs_c = goldtopmid[indices_c]
# extraindo a amostra da classe c
# caso deseje-se realizar amostragem com reposicao
# ou caso len(obs_c) < qtde_por_classe, pode-se
# informar o parametro replace=True
    amostra_c = obs_c.sample(qtde_por_classe)
# armazenando a amostra_c na lista de amostras
    amostras_por_classe.append(amostra_c)
# concatenando as amostras de cada classe em
# um único DataFrame
goldtopmid = pd.concat(amostras_por_classe)

from sklearn.utils import shuffle
goldtopmid = shuffle(goldtopmid)

"""#############Teste para gold20 > 0

goldpositivo = goldtopmid[goldtopmid['goldtop20'] > 0]
result = goldpositivo.iloc[:,-1]
result.value_counts().plot.bar()
goldpositivomid = goldtopmid[goldtopmid['goldmid20'] > 0]
result = goldpositivo.iloc[:,-1]
result.value_counts().plot.bar()

goldpositivoteste = goldpositivo[goldpositivo['goldmid20'] > 0]
result = goldpositivoteste.iloc[:,-1]
result.value_counts().plot.bar()
plt.xlabel('Vitorias(blue) top e mid na frente')

####balanceado off
FILTRO1=goldpositivoteste['bResult'] == 1
data1 = goldpositivoteste[FILTRO1]
FILTRO2=goldpositivoteste['bResult'] == 0 
data2 = goldpositivoteste[FILTRO2] 

from sklearn.utils import shuffle 
data1 = shuffle(data1) 
data1 = data1.sample(n = 500) 
data = data1.append(data2)

Xdata = data.iloc[:,0:36]
ydata = data.iloc[:,-1]

from imblearn.over_sampling import RandomOverSampler 
ros = RandomOverSampler(ratio='all')
X_res, y_res = ros.fit_sample(Xdata, ydata)

from sklearn.utils import shuffle
X_res = shuffle(X_res)
y_res = shuffle(y_res)"""


##########    KNN

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Xtopmid, ytopmid, test_size = 0.30)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=7)

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

###ARVORE DECISAO
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Xtopmid, ytopmid, test_size = 0.30)


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
X_train, X_test, y_train, y_test = train_test_split(Xtopmid, ytopmid, test_size = 0.30)

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

####################################

######################## Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


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




