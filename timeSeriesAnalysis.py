import os
from tqdm import tqdm, trange
import pandasql as sql
import pandas as pd

#the percantage deviation of pupil size at any given point in time is calculated
#from the average value of pupil size for either preceding pupil size or 
#pupil size, depending on what the time period is (check pupil.py to know the difference).

def createTimeSeriesPercentage(n):
    try:
        participant = str(n) if n > 9 else "0" + str(n)
        rawPupil = pd.read_csv("pupil/" + participant + ".csv")
        avgPupil = pd.read_csv("pupil_final/" + participant + "_Merged.csv")
        newTable = rawPupil[['Subject', 'Trial', 'Time']].copy()
        newTable['percentageSize'] = None
        for i in range(1, newTable.shape[0]):
            subject = int(rawPupil['Subject'][i])
            if 1 <= rawPupil['Time'][i] <=2:
                newTable['percentageSize'][i]  = rawPupil['Pupil'][i]/avgPupil['avg_pupil'][subject]
            elif -3 <= rawPupil['Time'][i] <= 0:
                newTable['percentageSize'][i]  = rawPupil['Pupil'][i]/avgPupil['avg_preceding_pupil'][subject]
        
        #DROP NONE VALUES AND SHIT
        newTable.to_csv("timeSeries/timeSeriesPercentage{}.csv".format(n))
    
    except FileNotFoundError:
        pass

def timeSeriesAccuracyTable(n):
    try:
        timeSeries = pd.read_csv("timeSeries/timeSeriesPercentage{}.csv".format(n))
        bigTable = pd.read_csv("bigTable/{}.csv".format(n))

        merged = sql.sqldf("select * from timeSeries, bigTable where timeSeries.Trial = bigTable.trial_index")
        merged = merged[[i for i in merged.columns if "unnamed" not in i.lower()]]

        merged.to_csv("timeSeries/timeSeriesBigTable{}.csv".format(n))

    except FileNotFoundError:
        pass

if __name__ == '__main__':
    import multiprocessing
    from pqdm.processes import pqdm
    import warnings
    warnings.filterwarnings('ignore')

    jobs = multiprocessing.cpu_count()
    # pqdm(range(1, 2), createTimeSeriesPercentage, n_jobs=jobs)
    pqdm(range(1, 100), timeSeriesAccuracyTable, n_jobs=jobs)