
#CALLING METHOD FOR SELECTION METHOD

def selectkfeature(indep,dep,n):
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    skb = SelectKBest(score_func=chi2,k=6)
    Kbest = skb.fit_transform(indep,dep)
    # This below process is to take the column name for model creation.
    mask = skb.get_support() #get_support() returns a boolean array where each element is True if the corresponding feature was selected and False otherwise.
    selected_features = indep.columns[mask] #The boolean mask is used to filter these column names, so only the names of the selected features are retained
    return Kbest,selected_features


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
    
def selectk_Classification(acclog,accsvml,accsvmnl,accknn,accdes,accrf): 
    
    dataframe=pd.DataFrame(index=['ChiSquare'],columns=['Logistic','SVMl','SVMnl','KNN','Decision','Random'])
    for number,idex in enumerate(dataframe.index):      
        dataframe['Logistic'][idex]=acclog[number]     
        dataframe['SVMl'][idex]=accsvml[number]
        dataframe['SVMnl'][idex]=accsvmnl[number]
        dataframe['KNN'][idex]=accknn[number]
        dataframe['Decision'][idex]=accdes[number]
        dataframe['Random'][idex]=accrf[number]
    return dataframe

