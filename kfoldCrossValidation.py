import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import multiprocessing
from pqdm.processes import pqdm
import pandasql as sql
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import pickle
from train_test_split import my_train_test_split

methods = ["decision_tree", "random_forest", "adaboost", "xgboost", "all"]
method_ = methods[0]

def test(participant, model):
    x_train, _, y_train, _, _ = my_train_test_split(participant=participant, test_percentage=0)
    predictions = model.predict(x_train)
    loss_array = abs(predictions - y_test)
    total = len(loss_array)

    incorrect = sum(loss_array)
    accuracy = (total - incorrect) / total

    #WRITE ACCURACY TO FILE OR SOMETHING

def crossVal(method, load_model=False):
    for i in trange(1,100):
        if not load_model:
            x_train, _, y_train, _, _ = my_train_test_split(participant=i, test_percentage=0)

        if method == "decision_tree":
            name = 'models/NonDeep/decisionTree{}.sav'.format(i)
            model = None if load_model else tree.DecisionTreeClassifier()
        elif method == "random_forest":
            name = 'models/NonDeep/randomForest{}.sav'.format(i)
            model = None if load_model else RandomForestClassifier(n_estimators=7, n_jobs=os.cpu_count())
        elif method == "adaboost":
            name = 'models/NonDeep/adaboost{}.sav'.format(i)
            model = None if load_model else AdaBoostClassifier(n_estimators=i, random_state=6)
        elif method == "xgboost":
            name = 'models/NonDeep/xgboost{}.sav'.format(participant)
            model = None if load_model else XGBClassifier(n_jobs=os.cpu_count(), random_state=6)

        if load_model:
            model = pickle.load(open(name, 'rb'))
        else:
            model.fit(x_train, y_train)
        
        pqdm(range(1,100), test, n_jobs=os.cpu_count())



if __name__ == '__main__':
    