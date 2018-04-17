import numpy as np
import dateutil.parser
from datetime import datetime

def time_to_real(hours, minutes, seconds):
    seconds_in_day = 24*60*60
    seconds = seconds + minutes*60 + hours*3600
    sin_time = np.sin(2*np.pi*seconds/seconds_in_day)
    cos_time = np.cos(2*np.pi*seconds/seconds_in_day)
    return cos_time, sin_time

def weekday_to_real(day):
    days_in_week = 7
    sin_time = np.sin(2*np.pi*day/days_in_week)
    cos_time = np.cos(2*np.pi*day/days_in_week)
    return cos_time, sin_time

def timestamp_to_features(timestamp):
    time = datetime.fromtimestamp(float(timestamp))
    weekday = time.weekday()
    cos_s, sin_s = time_to_real(time.hour, time.minute, time.second)
    cos_d, sin_d = weekday_to_real(weekday)
    return cos_s, sin_s, cos_d, sin_d

def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]
