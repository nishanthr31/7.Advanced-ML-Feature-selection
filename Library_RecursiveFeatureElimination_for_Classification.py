import pandas as pd
from sklearn.model_selection import train_test_split 
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import pickle
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier   
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


#CALLING METHOD FOR SELECTION METHOD

def rfeFeature(indep, dep,n):
    rfelist = []
    colnames_list = []
    log_model = LogisticRegression(solver='lbfgs')
    RF = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    DT = DecisionTreeClassifier(criterion='gini', max_features='sqrt', splitter='best', random_state=0)
    svc_model = SVC(kernel='linear', random_state=0)
    
    rfemodellist = [log_model, svc_model, RF, DT]
    
    for model in rfemodellist:
        print(model)
        log_rfe = RFE(estimator=model, n_features_to_select=n)
        log_fit = log_rfe.fit(indep, dep)
        log_rfe_feature = log_fit.transform(indep)
        rfelist.append(log_rfe_feature)

        # Get the column names selected by RFE
        selected_columns = [col for col, selected in zip(indep.columns, log_rfe.support_) if selected]
        colnames_list.append(selected_columns)
    return rfelist , colnames_list



#STEPS TO CREATE THE MACHINE LEARNING MODEL

def split_scalar(Kbest,dep):
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(Kbest,dep,test_size=0.3,random_state=0)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test, Y_train, Y_test

#FOR CLASSICIFIER

def accuracy(classifier,X_test,Y_test):
    Y_pred = classifier.predict(X_test)
    
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    from sklearn.metrics import accuracy_score
    
    cm = confusion_matrix(Y_test,Y_pred)
    clr = classification_report(Y_test,Y_pred)
    accuracy_value = accuracy_score(Y_test,Y_pred)
    return accuracy_value

#MODELS FOR CLASSIFICATION PROBLEM:

def LR(X_train, X_test,Y_train, Y_test):
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state = 0)
    classifier.fit(X_train, Y_train)
    acc = accuracy(classifier,X_test,Y_test)
    return acc

def Scvnonl(X_train, X_test,Y_train, Y_test):
    from sklearn.svm import SVC
    classifier = SVC(kernel = 'linear', random_state = 0)
    classifier.fit(X_train, Y_train)
    acc = accuracy(classifier,X_test,Y_test)
    return acc

def svclin(X_train, X_test,Y_train, Y_test):    
    from sklearn.svm import SVC
    classifier = SVC(kernel = 'rbf', random_state = 0)
    classifier.fit(X_train, Y_train)
    acc = accuracy(classifier,X_test,Y_test)
    return acc

def knn(X_train, X_test,Y_train, Y_test):    
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    classifier.fit(X_train, Y_train)
    acc = accuracy(classifier,X_test,Y_test)
    return acc

def DT(X_train, X_test,Y_train, Y_test):    
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    classifier.fit(X_train, Y_train)
    acc = accuracy(classifier,X_test,Y_test)
    return acc

def RF(X_train, X_test,Y_train, Y_test):    
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    classifier.fit(X_train, Y_train)
    acc = accuracy(classifier,X_test,Y_test)
    return acc
    
def rfe_classification(acclog,accsvmnl,accrf,accdes):
    rfedataframe=pd.DataFrame(index=['Logistic','SVC','Random','DecisionTree'],columns=['Logistic','SVMnl','Random','DecisionTree'])
    for number,idex in enumerate(rfedataframe.index):
        rfedataframe['Logistic'][idex]=acclog[number]
        rfedataframe["SVMnl"][idex] = accsvmnl[number]
        rfedataframe["Random"][idex] = accrf[number]
        rfedataframe["DecisionTree"][idex] = accdes[number]
        return rfedataframe
