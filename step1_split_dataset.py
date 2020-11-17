import time
import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

seed = 1388
time_start=time.time()
metadata = pd.read_csv("./metadata.txt", sep = "\t")
kf=KFold(n_splits=6,shuffle=True,random_state=seed)

for index,data in enumerate(kf.split(metadata.ID)):
    class_name = "class_" + str(index)
    print(class_name)
    
    train = data[0]
    test = data[1]
    
    train_meta = metadata.iloc[train]
    test_meta = metadata.iloc[test]
    
    train_per = train_meta.Res.value_counts()[1] / (train_meta.Res.value_counts()[1] + train_meta.Res.value_counts()[2])
    test_per = test_meta.Res.value_counts()[1] / (test_meta.Res.value_counts()[1] + test_meta.Res.value_counts()[2])
    
    print("Train sample NON / R: %.2f" % train_per)
    print("Test sample NON / R: %.2f" % test_per)
    
    tmp = {"A":test_meta.ID,"B":test_meta.ID}
    id_file = "./" + class_name + "/test_id.txt"
    pd.DataFrame(tmp).to_csv(id_file,sep="\t",index=False,header=None)
    
    command = "bash ./plink.sh " + class_name
    os.system(command)
    
    print("----------------------------")

time_end=time.time()

print("Finished")
print('Running time: %.2f s' % (time_end-time_start))