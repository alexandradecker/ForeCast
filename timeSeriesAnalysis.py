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
    