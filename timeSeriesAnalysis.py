import os
from tqdm import tqdm, trange
import pandasql as sql
import pandas as pd

#the percantage deviation of pupil size at any given point in time is calculate
#from the average value of pupil size for either preceding pupil size or 
#pupil size, depending on what the time period is (check pupil.py to know the difference).

def createTimeSeriesPercentage(n):
    participant = n if n > 9 else "0" + str(n)
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
    newTable.to_csv("newTable.csv") #DELETE THIS FILE
    print(newTable)


if __name__ == '__main__':
    createTimeSeriesPercentage(1)
    pass