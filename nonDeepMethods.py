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
participant = 1
method = methods[1]
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
        model = RandomForestClassifier(n_estimators=49, n_jobs=os.cpu_count())
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

# elif method == "adaboost":

#     if not test:
#         model = AdaBoostRegressor(n_estimators=935, random_state=6)
#         model.fit(x_train, y_train)

#         pickle.dump(model, open(
#             name, 'wb'))

#     else:
#         model = pickle.load(
#             open(name, 'rb'))
#         predictions = model.predict(x_test)
#         loss_array = (abs(predictions - y_test) / y_test) * 100

#         loss = sum(loss_array)/len(loss_array)

#         print("Prediction is within {}% of actual value".format(loss))

#     visualise = input("graph tree values? y/n: ")

#     if visualise.lower().strip() == 'y' or visualise.lower().strip() == 'yes':
#         loss_percentage = []
#         num_trees = [x for x in range(1, 1000)]
#         for i in trange(1, 1000):
#             model = AdaBoostRegressor(n_estimators=i)
#             model.fit(x_train, y_train)
#             predictions = model.predict(x_test)
#             loss_array = (abs(predictions - y_test) / y_test) * 100

#             loss = sum(loss_array)/len(loss_array)

#             loss_percentage.append(loss)

#         loss_percentage = np.array(loss_percentage)
#         num_trees = np.array(num_trees)
#         with open("loss_percentage_adaboost.txt", "w") as file:
#             np.savetxt(file, loss_percentage)

#         with open("num_trees_adaboost.txt", "w") as file:
#             np.savetxt(file, num_trees)

#         plt.plot(num_trees, loss_percentage)
#         plt.show()

#     # x = np.loadtxt("loss_percentage_adaboost.txt")
#     # y = np.loadtxt("num_trees_adaboost.txt")
#     # print(y[np.argmin(x)])

# # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


# elif method == "xgboost":

#     if not test:
#         model = XGBRegressor(n_jobs=os.cpu_count(),
#                              random_state=6)
#         model.fit(x_train, y_train)

#         pickle.dump(model, open(
#             name, 'wb'))

#     else:
#         model = pickle.load(
#             open(name, 'rb'))
#         predictions = model.predict(x_test)
#         loss_array = (abs(predictions - y_test) / y_test) * 100

#         loss = sum(loss_array)/len(loss_array)

#         print("Prediction is within {}% of actual value".format(loss))

# # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

# elif method == "all":
#     # use all and average results
#     if pred_num == 0:
#         xgboostname = 'models/NonDeep/xgBoostRegressor.sav'
#         adaboostname = 'models/NonDeep/adaBoostRegressor.sav'
#         randomforestname = 'models/NonDeep/randomForestRegressor.sav'
#         treename = 'models/NonDeep/decisionTreeRegressor.sav'

#     elif pred_num == 1:
#         xgboostname = 'models/NonDeep/xgBoostRegressorHigh.sav'
#         adaboostname = 'models/NonDeep/adaBoostRegressorHigh.sav'
#         randomforestname = 'models/NonDeep/randomForestRegressorHigh.sav'
#         treename = 'models/NonDeep/decisionTreeRegressorHigh.sav'

#     elif pred_num == 2:
#         xgboostname = 'models/NonDeep/xgBoostRegressorLow.sav'
#         adaboostname = 'models/NonDeep/adaBoostRegressorLow.sav'
#         randomforestname = 'models/NonDeep/randomForestRegressorLow.sav'
#         treename = 'models/NonDeep/decisionTreeRegressorLow.sav'

#     elif pred_num == 3:
#         xgboostname = 'models/NonDeep/xgBoostRegressorClose.sav'
#         adaboostname = 'models/NonDeep/adaBoostRegressorClose.sav'
#         randomforestname = 'models/NonDeep/randomForestRegressorClose.sav'
#         treename = 'models/NonDeep/decisionTreeRegressorClose.sav'

#     elif pred_num == 4:
#         xgboostname = 'models/NonDeep/xgBoostRegressorAdjClose.sav'
#         adaboostname = 'models/NonDeep/adaBoostRegressorAdjClose.sav'
#         randomforestname = 'models/NonDeep/randomForestRegressorAdjClose.sav'
#         treename = 'models/NonDeep/decisionTreeRegressorAdjClose.sav'

#     if not test:
#         print("Train each model separately, then test here!")

#     else:
#         xgboost = pickle.load(
#             open(xgboostname, 'rb'))
#         xgboost_predictions = xgboost.predict(x_test)

#         adaboost = pickle.load(
#             open(adaboostname, 'rb'))
#         adaboost_predictions = adaboost.predict(x_test)

#         randomforest = pickle.load(
#             open(randomforestname, 'rb'))
#         randomforest_predictions = randomforest.predict(x_test)

#         tree = pickle.load(
#             open(treename, 'rb'))
#         tree_predictions = tree.predict(x_test)

#         predictions = xgboost_predictions + adaboost_predictions + \
#             randomforest_predictions + tree_predictions
#         predictions /= 4

#         loss_array = (abs(predictions - y_test) / y_test) * 100

#         loss = sum(loss_array)/len(loss_array)

#         print("Prediction is within {}% of actual value".format(loss))
