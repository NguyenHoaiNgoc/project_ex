# 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

iris_data=load_iris()
features=iris_data.data
target=iris_data.target

features_train,features_test,target_train,target_test=train_test_split(features,target,test_size=0.2)
lrg=LogisticRegression()
lrg.fit(features_train,target_train)
score=lrg.score(features_test,target_test)

pickle.dump(lrg,open("iris_lrg_model01.pkl", 'wb'))