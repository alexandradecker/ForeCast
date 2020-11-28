'''
Split the final data into training, testing and PCA analysis sets.
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def my_train_test_split(participant):
    df = pd.read_csv("bigTable/"+str(participant)+".csv")
    df = df[['max(RIGHT_FIX_INDEX)', 'avg(RIGHT_IN_BLINK)', 'avg(RIGHT_IN_SACCADE)', 'avg(RIGHT_PUPIL_SIZE)', 'max_pupil', 'avg_preceding_pupil', 'avg_pupil', 'change_pupil',
             'rt', 'stimnum', 'acc', 'rtResidual', 'rtResidualC', 'rt_rec', 'trialno_rec', 'ratingConfidence', 'pupil_deviation', 'acc_rec']]
    df.dropna(inplace=True)
    data = np.array(df.values.tolist())

    x1 = data[1:, :-1]
    y1 = data[1:, -1]
    x = []
    y = []

    for i in range(len(x1)):
        if len(x1[i]) == 17:  # CHANGE VALUE DEPENDING ON NUM OF COLUMNS IN X
            x.append(x1[i])
            y.append(y1[i])
    x = np.asarray(x)
    y = np.asarray(y)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=6)

    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)

    for_pca = x[1:, :]

    for i in x_train:
        for j in i:
            if j is None:
                print(type(j))
                break
    return x_train, x_test, y_train, y_test, for_pca


if __name__ == "__main__":
    # x_train, x_test, y_train, y_test, for_pca = my_train_test_split()
    # print(len(x_train[0]))
    print(len(my_train_test_split(1)[0][1]))
    pass
