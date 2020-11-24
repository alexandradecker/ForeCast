'''
Split the final data into training, testing and PCA analysis sets.
'''
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split


def my_train_test_split(participant):
    df = pd.read_csv("bigTable/"+str(participant)+".csv")
    df = df[df.columns.tolist()[3:]]
    data = np.array(df.values.tolist())

    print(data)

    # x_train, x_test, y_train, y_test = train_test_split(
    #     x, y, test_size=0.3, random_state=6)

    # assert len(x_train) == len(y_train)
    # assert len(x_test) == len(y_test)

    # for_pca = data[:, 7:]
    # os.chdir("..")

    # return x_train.astype(float), x_test.astype(float), y_train.astype(float), y_test.astype(float), for_pca.astype(float)


if __name__ == "__main__":
    # x_train, x_test, y_train, y_test, for_pca = my_train_test_split()
    # print(len(x_train[0]))
    my_train_test_split(1)
