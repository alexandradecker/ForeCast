import pandas as pd
from tqdm import tqdm, trange
import pandasql as sql
import os


def avg_pupil_size(participant):
    data = pd.read_csv("pupil/" + participant + ".csv")
    cleaned = sql.sqldf(
        "select max(subject) as participant, Trial, avg(Pupil) as avg_pupil from data where Time >= 1 and Time <= 2 group by Trial")
    return cleaned


def avg_preceding_pupil_size(participant):
    data = pd.read_csv("pupil/" + participant + ".csv")
    cleaned = sql.sqldf(
        "select max(subject) as participant, Trial, avg(Pupil) as avg_preceding_pupil from data where Time >= -3 and Time <= 0 group by Trial")
    return cleaned


def max_pupil_size(participant):
    data = pd.read_csv("pupil/" + participant + ".csv")
    cleaned = sql.sqldf(
        "select max(subject) as participant, Trial, max(Pupil) as max_pupil from data where Time >= 0.5 and Time <= 2 group by Trial")
    cleaned["Time"] = "NULL"

    sizeCleaned = len(cleaned.index)
    sizeData = len(data.index)
    traversedIndex = 0
    for i in trange(sizeCleaned):
        max_pupil_size = cleaned["max_pupil"][i]
        for j in range(traversedIndex, sizeData):
            if data["Trial"][j] == i + 1 and data["Pupil"][j] == max_pupil_size:
                time = data["Time"][j]
                cleaned["Time"][i] = time
                traversedIndex = j
                break

    return cleaned


def merge(participant):
    avgPupilSize = pd.read_csv(
        "pupil_final/" + participant + "_AvgPupilSize.csv")
    avgPrecedingPupilSize = pd.read_csv(
        "pupil_final/" + participant + "_AvgPrecedingPupilSize.csv")
    maxPupilSize = pd.read_csv(
        "pupil_final/" + participant + "_MaxPupilSize.csv")

    merge = sql.sqldf(
        "select participant, Trial as trial_index, max_pupil, Time, avg_preceding_pupil, avg_pupil from avgPupilSize natural join avgPrecedingPupilSize natural join maxPupilSize")
    merge["change_pupil"] = merge["avg_pupil"] - merge["avg_preceding_pupil"]

    return merge


def delete_non_merged():
    os.chdir("pupil_final")
    for file in os.listdir():
        if not file.endswith("Merged.csv"):
            os.remove(file)


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    for i in trange(1, 100):
        try:
            if i < 10:
                participant = "0" + str(i)
            else:
                participant = str(i)

            avg_pupil_size(participant).to_csv(
                "pupil_final/" + participant + "_AvgPupilSize.csv")

            avg_preceding_pupil_size(participant).to_csv(
                "pupil_final/" + participant + "_AvgPrecedingPupilSize.csv")

            max_pupil_size(participant).to_csv("pupil_final/" +
                                               participant + "_MaxPupilSize.csv")

            merge(participant).to_csv("pupil_final/" +
                                      participant + "_Merged.csv")
        except Exception as e:
            print(e)

    delete_non_merged()
