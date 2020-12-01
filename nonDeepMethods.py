from train_test_split import my_train_test_split
import numpy as np
from tqdm import tqdm, trange
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import pickle
import os

methods = ["decision_tree", "random_forest", "adaboost", "xgboost", "all"]
participant = None
method = methods[2]
test = False

x_train, x_test, y_train, y_test, _ = my_train_test_split(participant)

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


if method == "decision_tree":

    name = 'models/NonDeep/decisionTree{}.sav'.format(participant)

    if not test:
        model = tree.DecisionTreeClassifier()
        model.fit(x_train, y_train)

        pickle.dump(model, open(
            name, 'wb'))

        visualise = input("save fig? y/n: ")
        if visualise.lower().strip() == 'y' or visualise.lower().strip() == 'yes':
            fig = plt.figure()
            _ = tree.plot_tree(model, filled=True)
            fig.savefig(
                'models/NonDeep/decisionTree{}.png'.format(participant))

    else:
        model = pickle.load(
            open(name, 'rb'))
        predictions = model.predict(x_test)
        loss_array = abs(predictions - y_test)
        total = len(loss_array)

        incorrect = sum(loss_array)
        accuracy = (total - incorrect) / total

        print("Accuracy = {}%".format(accuracy))


# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

elif method == "random_forest":
    name = 'models/NonDeep/randomForest{}.sav'.format(participant)

    if not test:
        model = RandomForestClassifier(n_estimators=7, n_jobs=os.cpu_count())
        model.fit(x_train, y_train)

        pickle.dump(model, open(
            name, 'wb'))

        visualise = input("graph tree values? y/n: ")
        if visualise.lower().strip() == 'y' or visualise.lower().strip() == 'yes':

            acc = []
            for i in trange(1, 1000):
                model = RandomForestClassifier(
                    n_estimators=i, n_jobs=os.cpu_count())
                model.fit(x_train, y_train)
                predictions = model.predict(x_test)
                loss_array = abs(predictions - y_test)

                total = len(loss_array)

                incorrect = sum(loss_array)
                accuracy = (total - incorrect) / total
                acc.append(accuracy)

            acc = np.asarray(acc)

            with open("models/NonDeep/acc_percentage_randomforest.txt", "w") as file:
                np.savetxt(file, acc)

            plt.plot(range(1, 1000), acc)
            plt.show()
        else:
            try:
                vals = np.loadtxt(
                    "models/NonDeep/acc_percentage_randomforest.txt", dtype=np.float64)
                print("Optimal Number of Trees: {}".format(np.argmax(vals) + 1))
            except Exception as e:
                print(e)
    else:
        model = pickle.load(
            open(name, 'rb'))
        predictions = model.predict(x_test)
        loss_array = abs(predictions - y_test)

        total = len(loss_array)

        incorrect = sum(loss_array)
        accuracy = (total - incorrect) / total

        print("Accuracy = {}%".format(accuracy))

# # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

elif method == "adaboost":

    name = 'models/NonDeep/adaboost{}.sav'.format(participant)

    if not test:
        model = AdaBoostClassifier(n_estimators=97, random_state=6)
        model.fit(x_train, y_train)

        pickle.dump(model, open(
            name, 'wb'))

        visualise = input("graph tree values? y/n: ")

        if visualise.lower().strip() == 'y' or visualise.lower().strip() == 'yes':
            acc = []
            for i in trange(1, 1000):
                model = AdaBoostClassifier(n_estimators=i, random_state=6)
                model.fit(x_train, y_train)
                predictions = model.predict(x_test)
                loss_array = abs(predictions - y_test)
                total = len(loss_array)
                incorrect = sum(loss_array)
                accuracy = (total - incorrect) / total
                acc.append(accuracy)

            acc = np.array(acc)
            with open("models/NonDeep/acc_percentage_adaboost.txt", "w") as file:
                np.savetxt(file, acc)

            plt.plot(range(1, 1000), acc)
            plt.show()
        else:
            try:
                vals = np.loadtxt(
                    "models/NonDeep/acc_percentage_adaboost.txt", dtype=np.float64)
                print("Optimal Number of Trees: {}".format(np.argmax(vals) + 1))
            except Exception as e:
                print(e)

    else:
        model = pickle.load(
            open(name, 'rb'))
        predictions = model.predict(x_test)
        loss_array = abs(predictions - y_test)

        total = len(loss_array)

        incorrect = sum(loss_array)
        accuracy = (total - incorrect) / total

        print("Accuracy = {}%".format(accuracy))


# # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


elif method == "xgboost":

    name = 'models/NonDeep/xgboost{}.sav'.format(participant)

    if not test:
        model = XGBClassifier(n_jobs=os.cpu_count(),
                              random_state=6)
        model.fit(x_train, y_train)

        pickle.dump(model, open(
            name, 'wb'))

    else:
        model = pickle.load(
            open(name, 'rb'))
        predictions = model.predict(x_test)
        loss_array = abs(predictions - y_test)

        total = len(loss_array)

        incorrect = sum(loss_array)
        accuracy = (total - incorrect) / total

        print("Accuracy = {}%".format(accuracy))

# # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

elif method == "all":
    # use all and average results

    xgboostname = 'models/NonDeep/xgboost{}.sav'.format(participant)
    adaboostname = 'models/NonDeep/adaboost{}.sav'.format(participant)
    randomforestname = 'models/NonDeep/randomForest{}.sav'.format(participant)
    treename = 'models/NonDeep/decisionTree{}.sav'.format(participant)

    if not test:
        print("Train each model separately, then test here!")

    else:
        xgboost = pickle.load(
            open(xgboostname, 'rb'))
        xgboost_predictions = xgboost.predict(x_test)

        adaboost = pickle.load(
            open(adaboostname, 'rb'))
        adaboost_predictions = adaboost.predict(x_test)

        randomforest = pickle.load(
            open(randomforestname, 'rb'))
        randomforest_predictions = randomforest.predict(x_test)

        tree = pickle.load(
            open(treename, 'rb'))
        tree_predictions = tree.predict(x_test)

        predictions = xgboost_predictions + adaboost_predictions + \
            randomforest_predictions + tree_predictions

        for i in range(len(predictions)):
            if predictions[i] >= 2:
                predictions[i] = 1
            else:
                predictions[i] = 0

        loss_array = abs(predictions - y_test)

        total = len(loss_array)

        incorrect = sum(loss_array)
        accuracy = (total - incorrect) / total

        print("Accuracy = {}%".format(accuracy))
