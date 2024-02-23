# das kann ignoriert werde, war lediglich für den test des imports gedacht

def add(a,b):
    return(a+b)

def multiply(a,b):
    return(a*b)

# Ab hier relevant

from darts import TimeSeries

# Transform pandas object to darts TimeSeries format
from darts import TimeSeries
import pandas as pd

def transform_to_darts_format(pandas_object):
    pandas_object_copy = pandas_object.copy() # otherwise it changes the index globally
    pandas_object_copy.index = pandas_object_copy.index.to_timestamp()
    
    if isinstance(pandas_object_copy, pd.DataFrame):
        darts_ts = TimeSeries.from_dataframe(pandas_object_copy)
        
    elif isinstance(pandas_object_copy, pd.Series):
        darts_ts = TimeSeries.from_series(pandas_object_copy)
    return darts_ts
