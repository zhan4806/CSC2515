
import logging
logging.basicConfig()
import numpy as np
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from scipy.stats.stats import pearsonr
from sklearn.metrics.cluster import normalized_mutual_info_score
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier #Decision Tree Classifier
from sklearn.svm import SVC #SVM Classifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import Imputer
import pdb
import os

file = open('Arrhythmia_8_3.txt','w') 
script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
arr_path = "Classification Dataset/Arrhythmia/arrhythmia.data"
test_ratio=0.2
logger=logging.getLogger()

def load_data(rel_path, feature_number):    
    abs_file_path = os.path.join(script_dir, rel_path)
    
    X=[]
    y=[]
    
    sparse_count=0
    with open(abs_file_path,'r') as f:
        lines=[x.strip() for x in f.readlines()]
    for line in lines:
        elements=line.split(',')
        for i in range(0,len(elements)):
            if elements[i]== '?':
                elements[i]=np.NaN
                sparse_count=sparse_count+1
            else:
                elements[i]=float(elements[i])
        X.append(elements[0:feature_number]) 
        y.append(elements[feature_number])
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    X=imp.fit_transform(X)
    unique_elements, counts_elements = np.unique(y, return_counts=True)
    for i in range(len(unique_elements)):
        count=counts_elements[i]
        y_value=unique_elements[i]
        if count <5:
            y_index=np.where(y==y_value)
            X=np.delete(X,y_index,0)
            y=np.delete(y,y_index,None)
    X_train, X_test, y_train, y_test = train_test_split(np.matrix(X), y, test_size=test_ratio)
    file.write("Train data size is: {}".format(X_train.shape[0])+ "\n")
    file.write("Test data size is: {}".format(X_test.shape[0])+ "\n")
    file.write("Original feature amount is: {}".format(X_train.shape[1])+ "\n")
    data_size=(np.shape(X)[0]+1)*(np.shape(X)[1]+1)
    file.write("Sparse value rate is: {}".format(sparse_count/data_size)+"\n")
    
    return X_train, y_train, X_test, y_test

def calcualte_pcc(train_data,train_target):
    corr=[]
    for  i in range(train_data.shape[1]):
        corr.append((pearsonr(train_data[:,i], np.matrix(train_target).T))[0][0])

    return np.nan_to_num(corr)

def filter_pcc(train_data,train_target,t):
    index=[]
    corr=calcualte_pcc(train_data,train_target)
    for  i in range(len(corr)):
        if(np.absolute(corr[i])>=t):
            index.append(i)
    return index

def calculate_mi(train_data,train_target):
    mi=[]
    for  i in range(train_data.shape[1]):
        mi.append(normalized_mutual_info_score(train_target,np.array(train_data[:,i]).flatten()))
    return mi

def filter_mi(train_data,train_target,t):
    index=[]
    mi=calculate_mi(train_data,train_target)
    for  i in range(train_data.shape[1]):
        if(mi[i]>=t):
            index.append(i)
    return index

def svm_classifier(train_data, train_labels, test_data, test_labels):
    C_range = 10. ** np.arange(-2,3)
    gamma_range = ['auto']
    param_grid = dict(gamma=gamma_range, C=C_range)
    
    grid_search = GridSearchCV(SVC(), param_grid=param_grid, cv=5)
    grid_search.fit(train_data, train_labels)

    model=grid_search.best_estimator_
    
    file.write("The best classifier is: {}".format(grid_search.best_estimator_)+ "\n")

    #evaluate the best SVM model
    train_pred = model.predict(train_data)
    file.write('SVM Classifier train accuracy = {}'.format((train_pred == train_labels).mean())+ "\n")
    file.write('SVM Classifier train loss = {}'.format((train_pred != train_labels).mean())+ "\n")
    test_pred = model.predict(test_data)
    file.write('SVM Classifier test accuracy = {}'.format((test_pred == test_labels).mean())+ "\n")
    file.write('SVM Classifier test loss = {}'.format((test_pred != test_labels).mean())+ "\n")
        
    return model

def sbs(train_data, train_target,test_data,test_target):
    floor=1
    ceil=train_data.shape[1]
    C_range = 10. ** np.arange(-2,3)
    gamma_range = ['auto']
    scores=[]
    indices=[]
    svm_models=[]
    for c in C_range:
        for g in gamma_range:
            svm_model=SVC(C=c,gamma=g)
            sfs1 = SFS(svm_model, 
                   k_features=(floor,ceil), 
                   forward=False, 
                   floating=False, 
                   verbose=0,
                   scoring='accuracy',
                   cv=5)
            sfs1 = sfs1.fit(train_data, train_target)
            scores.append(sfs1.k_score_)
            indices.append(sfs1.k_feature_idx_)
            svm_models.append(svm_model)

    best_model=np.argmax(scores)
    model=svm_models[best_model]
    file.write("The best Sequential Backward Elimination model contains {} features. Best model is {}.".format(len(indices[best_model]),model)+ "\n")

    model.fit(train_data[:,indices[best_model]],train_target)
    #evaluate the best model
    train_pred = model.predict(train_data[:,indices[best_model]])
    file.write('SVM Classifier train accuracy = {}'.format((train_pred == train_target).mean())+ "\n")
    file.write('SVM Classifier train loss = {}'.format((train_pred != train_target).mean())+ "\n")
    test_pred = model.predict(test_data[:,indices[best_model]])
    file.write('SVM Classifier test accuracy = {}'.format((test_pred == test_target).mean())+ "\n")
    file.write('SVM Classifier test loss = {}'.format((test_pred != test_target).mean())+ "\n")

    return model

def sbfs(train_data, train_target,test_data,test_target): 
    floor=1
    ceil=train_data.shape[1]  
    C_range = 10. ** np.arange(-2,3)
    gamma_range = ['auto']
    scores=[]
    indices=[]
    svm_models=[]
    for c in C_range:
        for g in gamma_range:
            svm_model=SVC(C=c,gamma=g)
            sfs1 = SFS(svm_model, 
                   k_features=(floor,ceil), 
                   forward=False, 
                   floating=True, 
                   verbose=0,
                   scoring='accuracy',
                   cv=5)
            sfs1 = sfs1.fit(train_data, train_target)
            scores.append(sfs1.k_score_)
            indices.append(sfs1.k_feature_idx_)
            svm_models.append(svm_model)

    best_model=np.argmax(scores)
    model=svm_models[best_model]
    file.write("The best Sequential Backward Floating Selection model contains {} features. Best model is {}.".format(len(indices[best_model]),model)+ "\n")

    model.fit(train_data[:,indices[best_model]],train_target)
    #evaluate the best model
    train_pred = model.predict(train_data[:,indices[best_model]])
    file.write('SVM Classifier train accuracy = {}'.format((train_pred == train_target).mean())+ "\n")
    file.write('SVM Classifier train loss = {}'.format((train_pred != train_target).mean())+ "\n")
    test_pred = model.predict(test_data[:,indices[best_model]])
    file.write('SVM Classifier test accuracy = {}'.format((test_pred == test_target).mean())+ "\n")
    file.write('SVM Classifier test loss = {}'.format((test_pred != test_target).mean())+ "\n")

    return model

def main():
    train_data, train_target,test_data,test_target=load_data(arr_path,279)

    # Hybrid: MI + SBFS   
    file.write("Hybrid: MI + SBFS starts:"+"\n")
    for t in np.arange(0.3, 0.4,0.1):
        start_time = time.time()
        index=filter_mi(train_data,train_target,t)
        file.write("Mutual Information threshold is: {}, {} features left.".format(t,len(index))+ "\n")
        if len(index) != 0:
            sbfs(train_data[:,index], train_target,test_data[:,index],test_target)
        file.write("Execution time is: %s seconds." % (time.time() - start_time)+ "\n"+"\n")
    
    file.close() 

    
if __name__ == '__main__':    
    try:
        main()
    except Exception:
        logger.warn()

