import time
import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import sklearn.model_selection as ms
from sklearn.ensemble import RandomForestClassifier

model_list = []

for i in range(0,6):
        class_name = "class_6_1388_" + str(i)
        metadata = pd.read_csv("./metadata.txt", sep = "\t")
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

        # calculate random forest importance score
        param_grid = {'n_estimators':np.arange(0,200,10),
                'max_depth':np.arange(1, 20, 1)}

        rfc = RandomForestClassifier(random_state=90)

        gs = ms.GridSearchCV(rfc,param_grid,cv=10,n_jobs=-1)
        gs.fit(X_train, y_train)
        rfc_model_best = gs.best_estimator_
        feat_importances = pd.Series(rfc_model_best.feature_importances_, index=snp)
        
        imp_coef = pd.DataFrame(feat_importances.nlargest(1500))
        imp_index = imp_coef.index
        model_list.append(rfc_model_best)

        imp_coef_file = "./" + class_name + "/top1500_coef.txt"
        pd.DataFrame(imp_coef).to_csv(imp_coef_file,sep="\t",index=True,header=None)
        print("finish")

pd.DataFrame(model_list).to_csv("random_forest_fs_model_para.txt")