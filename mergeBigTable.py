import os
from tqdm import tqdm, trange
import pandasql as sql
import pandas as pd


def mergeBigTable(i):
    if not os.path.isdir("bigTable"):
        os.mkdir("bigTable")

    num = str(i) if i > 9 else "0" + str(i)

    eyetracking = pd.read_csv(
        "eyetracking_final/" + num + "_cleaned.csv")
    pupil = pd.read_csv("pupil_final/" + num + "_Merged.csv")
    behavioural = pd.read_csv(
        "behavioral/" + str(i) + "_behavioral.csv")

    bigBoi = sql.sqldf(
        'select * from eyetracking natural join pupil natural join behavioural')

    bigBoi[bigBoi.columns.tolist()[1:]].to_csv(
        "bigTable/" + str(i) + ".csv")


if __name__ == "__main__":

    try:
        for i in trange(1, 100):
            mergeBigTable(i)
    except Exception as e:
        print(e)
