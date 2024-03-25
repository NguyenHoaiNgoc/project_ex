from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import pickle

iris = load_iris()


features, target = iris.data, iris.target
features_train,features_test,target_train,target_test=train_test_split(features,target,test_size=0.2)
lrg=LogisticRegression()
lrg.fit(features_train,target_train)

score=lrg.score(features_test,target_test)
print(score)

pickle.dump(lrg,open("iris_clf_lrg_model.pkl", 'wb'))