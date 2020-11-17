import time
import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import sklearn.model_selection as ms
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import roc_auc_score as AUC
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import pickle

def svm_best(train_X, train_y, testx, testy, class_name, num):
    svm_train_test = []
    params = [
            {'kernel': ['linear'], 'C': [1, 10, 100, 100]},
            {'kernel': ['poly'], 'C': [1], 'degree': [2, 3]},
            {'kernel': ['rbf'], 'C': [1, 10, 100, 100], 'gamma':[1, 0.1, 0.01, 0.001]}
            ]

    model = ms.GridSearchCV(SVC(probability=True), 
                            params, 
                            refit=True,
                            return_train_score=True,
                            cv=KFold(train_X.shape[0]))

    model.fit(train_X, train_y)
    model_best = model.best_estimator_
    file_name = class_name + "_rem_top" + str(num) + "svm.pickle"
    with open(file_name, 'wb') as f:
        pickle.dump(model_best, f)

    train_auc = cross_val_score(model_best, train_X, train_y, scoring='roc_auc', cv=10).mean()  
    train_precision = cross_val_score(model_best, train_X, train_y, scoring='precision', cv=10).mean()
    train_recall = cross_val_score(model_best, train_X, train_y, scoring='recall', cv=10).mean()
    train_acc = cross_val_score(model_best, train_X, train_y, scoring='accuracy', cv=10).mean()

    svm_train_test.append(train_precision)
    svm_train_test.append(train_recall)
    svm_train_test.append(train_acc)
    svm_train_test.append(train_auc)

    precision = precision_score(testy, model_best.predict(testx))
    recall = recall_score(testy, model_best.predict(testx))
    acc = accuracy_score(testy, model_best.predict(testx))
    auc = AUC(testy,model_best.predict(testx))

    svm_train_test.append(precision)
    svm_train_test.append(recall)
    svm_train_test.append(acc)
    svm_train_test.append(auc)

    return svm_train_test,model.best_params_

def rf_best(train_X, train_y, testx, testy, class_name, num):
    rf_train_test = []
    params = {'n_estimators':np.arange(0,200,10),
                 'max_depth':np.arange(1, 20, 1)}

    model = ms.GridSearchCV(RandomForestClassifier(random_state=90), 
                            params, 
                            refit=True,
                            return_train_score=True,
                            cv=10,n_jobs=-1)

    model.fit(train_X, train_y)
    model_best = model.best_estimator_
    file_name = class_name + "_rem_top" + str(num) + "rf.pickle"
    with open(file_name, 'wb') as f:
        pickle.dump(model_best, f)

    train_auc = cross_val_score(model_best, train_X, train_y, scoring='roc_auc', cv=10).mean()  
    train_precision = cross_val_score(model_best, train_X, train_y, scoring='precision', cv=10).mean()
    train_recall = cross_val_score(model_best, train_X, train_y, scoring='recall', cv=10).mean()
    train_acc = cross_val_score(model_best, train_X, train_y, scoring='accuracy', cv=10).mean()

    rf_train_test.append(train_precision)
    rf_train_test.append(train_recall)
    rf_train_test.append(train_acc)
    rf_train_test.append(train_auc)

    precision = precision_score(testy, model_best.predict(testx))
    recall = recall_score(testy, model_best.predict(testx))
    acc = accuracy_score(testy, model_best.predict(testx))
    auc = AUC(testy,model_best.predict(testx))

    rf_train_test.append(precision)
    rf_train_test.append(recall)
    rf_train_test.append(acc)
    rf_train_test.append(auc)

    return rf_train_test,model.best_params_

def knn_best(train_X, train_y, testx, testy, class_name, num):
    knn_train_test = []
    params = [
        {
            'weights':['uniform'],
            'n_neighbors':[i for i in range(1,11)]
        },
        {
            'weights':['distance'],
            'n_neighbors':[i for i in range(1,11)],
            'p':[i for i in range(1,6)]
        }
    ]

    model = ms.GridSearchCV(KNeighborsClassifier(), 
                            params, 
                            refit=True,
                            return_train_score=True,
                            cv=10,n_jobs=-1)

    model.fit(train_X, train_y)
    model_best = model.best_estimator_
    file_name = class_name + "_rem_top" + str(num) + "knn.pickle"
    with open(file_name, 'wb') as f:
        pickle.dump(model_best, f)

    train_auc = cross_val_score(model_best, train_X, train_y, scoring='roc_auc', cv=10).mean()  
    train_precision = cross_val_score(model_best, train_X, train_y, scoring='precision', cv=10).mean()
    train_recall = cross_val_score(model_best, train_X, train_y, scoring='recall', cv=10).mean()
    train_acc = cross_val_score(model_best, train_X, train_y, scoring='accuracy', cv=10).mean()

    knn_train_test.append(train_precision)
    knn_train_test.append(train_recall)
    knn_train_test.append(train_acc)
    knn_train_test.append(train_auc)

    precision = precision_score(testy, model_best.predict(testx))
    recall = recall_score(testy, model_best.predict(testx))
    acc = accuracy_score(testy, model_best.predict(testx))
    auc = AUC(testy,model_best.predict(testx))

    knn_train_test.append(precision)
    knn_train_test.append(recall)
    knn_train_test.append(acc)
    knn_train_test.append(auc)

    return knn_train_test,model.best_params_

def lr_best(train_X, train_y, testx, testy, class_name, num):
    lr_train_test = []
    params = {'C':[0.01,0.1,1,10],
    'class_weight':[None,"balanced"],
    'penalty':['l1', 'l2', None]
    }

    model = ms.GridSearchCV(LogisticRegression(max_iter = 10000, random_state=90), 
                            params, 
                            refit=True,
                            return_train_score=True,
                            cv=10,n_jobs=-1)

    model.fit(train_X, train_y)
    model_best = model.best_estimator_
    file_name = class_name + "_rem_top" + str(num) + "lr.pickle"
    with open(file_name, 'wb') as f:
        pickle.dump(model_best, f)

    train_auc = cross_val_score(model_best, train_X, train_y, scoring='roc_auc', cv=10).mean()  
    train_precision = cross_val_score(model_best, train_X, train_y, scoring='precision', cv=10).mean()
    train_recall = cross_val_score(model_best, train_X, train_y, scoring='recall', cv=10).mean()
    train_acc = cross_val_score(model_best, train_X, train_y, scoring='accuracy', cv=10).mean()

    lr_train_test.append(train_precision)
    lr_train_test.append(train_recall)
    lr_train_test.append(train_acc)
    lr_train_test.append(train_auc)

    precision = precision_score(testy, model_best.predict(testx))
    recall = recall_score(testy, model_best.predict(testx))
    acc = accuracy_score(testy, model_best.predict(testx))
    auc = AUC(testy,model_best.predict(testx))

    lr_train_test.append(precision)
    lr_train_test.append(recall)
    lr_train_test.append(acc)
    lr_train_test.append(auc)

    return lr_train_test,model.best_params_

def dt_best(train_X, train_y, testx, testy, class_name, num):
    dt_train_test = []
    params = {'criterion':["gini", "entropy"],
    'max_depth':np.arange(1, 20, 1),
    'min_samples_leaf':np.arange(1, 1+10, 1),
    'min_samples_split':np.arange(2, 2+20, 1),
    'class_weight':[None,"balanced"]
    }
              
    model = ms.GridSearchCV(DecisionTreeClassifier(random_state=90), 
                            params, 
                            refit=True,
                            return_train_score=True,
                            cv=10,n_jobs=-1)

    model.fit(train_X, train_y)
    model_best = model.best_estimator_
    file_name = class_name + "_rem_top" + str(num) + "dt.pickle"
    with open(file_name, 'wb') as f:
        pickle.dump(model_best, f)

    train_auc = cross_val_score(model_best, train_X, train_y, scoring='roc_auc', cv=10).mean()  
    train_precision = cross_val_score(model_best, train_X, train_y, scoring='precision', cv=10).mean()
    train_recall = cross_val_score(model_best, train_X, train_y, scoring='recall', cv=10).mean()
    train_acc = cross_val_score(model_best, train_X, train_y, scoring='accuracy', cv=10).mean()

    dt_train_test.append(train_precision)
    dt_train_test.append(train_recall)
    dt_train_test.append(train_acc)
    dt_train_test.append(train_auc)

    precision = precision_score(testy, model_best.predict(testx))
    recall = recall_score(testy, model_best.predict(testx))
    acc = accuracy_score(testy, model_best.predict(testx))
    auc = AUC(testy,model_best.predict(testx))

    dt_train_test.append(precision)
    dt_train_test.append(recall)
    dt_train_test.append(acc)
    dt_train_test.append(auc)

    return dt_train_test,model.best_params_

def en_best(train_X, train_y, testx, testy, class_name, num):
    en_train_test = []
    params = {'C':[0.01,0.1,1,10],
    'class_weight':[None,"balanced"]
    }

    model = ms.GridSearchCV(LogisticRegression(max_iter = 10000, random_state=90,penalty = 'elasticnet',solver='saga',l1_ratio=0.5), 
                            params, 
                            refit=True,
                            return_train_score=True,
                            cv=10,n_jobs=-1)

    model.fit(train_X, train_y)
    model_best = model.best_estimator_
    file_name = class_name + "_rem_top" + str(num) + "en.pickle"
    with open(file_name, 'wb') as f:
        pickle.dump(model_best, f)

    train_auc = cross_val_score(model_best, train_X, train_y, scoring='roc_auc', cv=10).mean()  
    train_precision = cross_val_score(model_best, train_X, train_y, scoring='precision', cv=10).mean()
    train_recall = cross_val_score(model_best, train_X, train_y, scoring='recall', cv=10).mean()
    train_acc = cross_val_score(model_best, train_X, train_y, scoring='accuracy', cv=10).mean()

    en_train_test.append(train_precision)
    en_train_test.append(train_recall)
    en_train_test.append(train_acc)
    en_train_test.append(train_auc)

    precision = precision_score(testy, model_best.predict(testx))
    recall = recall_score(testy, model_best.predict(testx))
    acc = accuracy_score(testy, model_best.predict(testx))
    auc = AUC(testy,model_best.predict(testx))

    en_train_test.append(precision)
    en_train_test.append(recall)
    en_train_test.append(acc)
    en_train_test.append(auc)

    return en_train_test,model.best_params_

mean_AUC = []

for num in (5,10,20,30,40,50,100,200,500,1000,1500):
    seed = 1388
    n_split = 6
    metadata = pd.read_csv("./metadata.txt", sep = "\t")
    time_start=time.time()
    model_list = []
    svm_result = []
    rf_result = []
    knn_result = []
    lr_result = []
    dt_result = []
    en_result = []

    for i in range(0,n_split):
        tmp_model_list = []
        class_name = "class_" + str(n_split) + "_" + str(seed) + "_" + str(i)
        
        train_file = "./" + class_name + "/train.clump.genotype.txt"
        train_data = pd.read_csv(train_file, sep = " ")
        train_data = pd.merge(metadata,train_data,how="inner",left_on="ID",right_on="ID")
        train_data = train_data.dropna(axis=1, how='all')

        test_file = "./" + class_name + "/test.clump.genotype.txt"
        test_data = pd.read_csv(test_file, sep = " ")
        test_data = pd.merge(metadata,test_data,how="inner",left_on="ID",right_on="ID")
        test_data = test_data.dropna(axis=1, how='all')

        snp = train_data.columns.values
        snp = snp[3:]
        X_train = train_data[snp].values
        y_train = train_data['Res'].values

        X_test = test_data[snp].values
        y_test = test_data['Res'].values

        # select top N SNPs to build downstream ML models
        feat_file = "./" + class_name + "/top1500_coef.txt"
        feat_importances = pd.read_table(feat_file,header=None)
        feat_importances.columns=['snp','imp']
        imp_coef = feat_importances.sort_values(by="imp",ascending=False).head(num)
        imp_index = imp_coef['snp']

        train_X = train_data[imp_index].values
        train_y = train_data['Res'].values

        testx = test_data[imp_index].values
        testy = test_data['Res'].values

        svm_train_test,svm_model_best = svm_best(train_X, train_y, testx, testy)
        rf_train_test,rf_model_best = rf_best(train_X, train_y, testx, testy)
        knn_train_test,knn_model_best = knn_best(train_X, train_y, testx, testy)
        lr_train_test,lr_model_best = lr_best(train_X, train_y, testx, testy)
        dt_train_test,dt_model_best = dt_best(train_X, train_y, testx, testy)
        en_train_test,en_model_best = en_best(train_X, train_y, testx, testy)

        tmp_model_list.append(svm_model_best)
        tmp_model_list.append(rf_model_best)
        tmp_model_list.append(knn_model_best)
        tmp_model_list.append(lr_model_best)
        tmp_model_list.append(dt_model_best)
        tmp_model_list.append(en_model_best)
        
        model_list.append(tmp_model_list)
        
        svm_result.append(svm_train_test)
        rf_result.append(rf_train_test)
        knn_result.append(knn_train_test)
        lr_result.append(lr_train_test)
        dt_result.append(dt_train_test)
        en_result.append(en_train_test)

        print("Finished " + class_name)

    svm_result = pd.DataFrame(svm_result,columns=["Train_Precison","Train_Recall","Train_Accuracy","Train_AUC","Test_Precison","Test_Recall","Test_Accuracy","Test_AUC"])
    rf_result = pd.DataFrame(rf_result,columns=["Train_Precison","Train_Recall","Train_Accuracy","Train_AUC","Test_Precison","Test_Recall","Test_Accuracy","Test_AUC"])
    knn_result = pd.DataFrame(knn_result,columns=["Train_Precison","Train_Recall","Train_Accuracy","Train_AUC","Test_Precison","Test_Recall","Test_Accuracy","Test_AUC"])
    lr_result = pd.DataFrame(lr_result,columns=["Train_Precison","Train_Recall","Train_Accuracy","Train_AUC","Test_Precison","Test_Recall","Test_Accuracy","Test_AUC"])
    dt_result = pd.DataFrame(dt_result,columns=["Train_Precison","Train_Recall","Train_Accuracy","Train_AUC","Test_Precison","Test_Recall","Test_Accuracy","Test_AUC"])
    en_result = pd.DataFrame(en_result,columns=["Train_Precison","Train_Recall","Train_Accuracy","Train_AUC","Test_Precison","Test_Recall","Test_Accuracy","Test_AUC"])

    all_data = pd.concat([svm_result,rf_result,knn_result,lr_result,dt_result,en_result])
    all_data_file = str(num) + ".txt"
    all_data.to_csv(all_data_file,sep="\t",index=True,header=True)
    model_file = "models_top" + str(num) + ".txt"
    pd.DataFrame(model_list).to_csv(model_file,sep="\t",index=True,header=True)
    AUC_mean = all_data.Test_AUC.mean()
    mean_AUC.append(AUC_mean)
    print(num)