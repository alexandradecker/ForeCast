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


def work(i):
    try:
        mergeBigTable(i)

    except Exception as e:
        print(e)


def addMeanResponseTime(participant):
    try:
        num = str(participant)
        table = pd.read_csv("bigTable/" + num + ".csv")
        total_pupil_size = table["avg_pupil"].sum()
        num_rows = table["avg_pupil"].shape[0]
        avg_pupil_size = total_pupil_size / num_rows
        pupil_deviation = table["avg_pupil"] - avg_pupil_size
        # pd.concat([table, pupil_deviation], axis=1, ignore_index=True)
        # table = table[table.columns.tolist()[1:]]
        table["pupil_deviation"] = pupil_deviation

        table.to_csv("bigTable/" + num + ".csv")
    except Exception as e:
        print(e)


if __name__ == "__main__":

    import multiprocessing
    from pqdm.processes import pqdm
    import warnings
    warnings.filterwarnings('ignore')

    jobs = multiprocessing.cpu_count()
    pqdm(range(1, 100), work, n_jobs=jobs)
    pqdm(range(1, 100), addMeanResponseTime, n_jobs=jobs)
    pass
