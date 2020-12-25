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


def addPupilDeviation(participant):
    try:
        num = str(participant)
        table = pd.read_csv("bigTable/" + num + ".csv")
        total_pupil_size = table["avg_pupil"].sum()
        num_rows = table["avg_pupil"].shape[0]
        avg_pupil_size = total_pupil_size / num_rows
        pupil_deviation = table["avg_pupil"] - avg_pupil_size
        table["pupil_deviation"] = pupil_deviation
        table["pupil_deviation_percentage"] = pupil_deviation/avg_pupil_size

        table.to_csv("bigTable/" + num + ".csv")
    except Exception as e:
        print(e)


def addResponseDeviation(participant):
    try:
        num = str(participant)
        table = pd.read_csv("bigTable/" + num + ".csv")
        total_pupil_size = table["rt"].sum()
        num_rows = table["rt"].shape[0]
        avg_rt = total_pupil_size / num_rows
        rt_deviation = table["rt"] - avg_rt
        table["rt_deviation"] = rt_deviation
        table["rt_deviation_percentage"] = rt_deviation/avg_rt

        table.to_csv("bigTable/" + num + ".csv")
    except Exception as e:
        print(e)


def aggregateBigTables():
    table = pd.read_csv("bigTable/1.csv")

    for i in range(2, 100):
        try:
            next_ = pd.read_csv("bigTable/{}.csv".format(i))
            table.append(next_)
        except Exception as e:
            print(e)

    table.to_csv("bigTable/allParticipantsTable.csv")


if __name__ == "__main__":

    import multiprocessing
    from pqdm.processes import pqdm
    import warnings
    warnings.filterwarnings('ignore')

    jobs = multiprocessing.cpu_count()
    pqdm(range(1, 100), work, n_jobs=jobs)
    pqdm(range(1, 100), addPupilDeviation, n_jobs=jobs)
    pqdm(range(1, 100), addResponseDeviation, n_jobs=jobs)
    aggregateBigTables()
    pass
