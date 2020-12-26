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
from functools import partial
import os

methods = ["decision_tree", "random_forest", "adaboost", "xgboost", "all"]
method_ = methods[0]

def test(model, item):
    try:
        x_train, _, y_train, _, _ = my_train_test_split(participant=item, test_percentage=1)
        predictions = model.predict(x_train)
        loss_array = abs(predictions - y_train)
        total = len(loss_array)

        incorrect = sum(loss_array)
        accuracy = (total - incorrect) / total

        return accuracy
    except Exception as e:
        pass


def crossVal(method, load_model=False):
    for i in trange(1,100):
        try:
            if not load_model:
                x_train, _, y_train, _, _ = my_train_test_split(participant=i, test_percentage=1)
        except:
            continue

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
            try:
                model.fit(x_train, y_train) #THIS TRY EXCEPT BLOCK SHOULDNT BE THERE BUT THE 14th PARTICIPANT DATA IS INCONSISTENT
            except:
                continue
        
        func = partial(test, model)
        accuracy = pqdm(range(1,100), func, n_jobs=os.cpu_count())
        accuracy = np.array([a for a in accuracy if a])
      
        with open("testResults/{}.txt".format(i), 'w') as f:
            if len(accuracy) > 0:
                f.write("Accuracy: " + str(np.average(accuracy)) + "\n")
                f.write("Variance: " + str(np.var(accuracy)) + "\n")
                f.write("Standard Deviation: " + str(np.std(accuracy)) + "\n\n")
                for line in accuracy:
                    f.write(str(line) + "\n")



if __name__ == '__main__':
    crossVal(method_)