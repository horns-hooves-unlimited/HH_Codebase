def hh_create_bus_dates(date_type = 'string', begin_date = '1900-01-01', end_date = '2018-12-31', interval = 'day'):
# Version 0.02
# Creating the business dates list in form of pd.DatetimeIndex
# date_type - type of first and last dates representation: 'string' = '%Y-%m-%d'; 'date' = datetime.date
# begin_date - first date of dates list
# begin_date - last date of dates list
# interval - period of dates list: 'day' = days; 'week' = week ends; 'month' = month ends; 'quarter' = quarter ends; 'year' = year ends

    import pandas as pd # For date_range functionality
    from datetime import date # For date/string converting functionality
    
    date_format = '%Y-%m-%d'
    
    if date_type == 'string': # Converting dates to common string format in case of date formatting
        range_begin_date = begin_date
        range_end_date = end_date
    else:
        range_begin_date = begin_date.strftime(date_format)
        range_end_date = end_date.strftime(date_format)        
        
    bus_freq_dict = {'day': 'B', 'week': 'W-FRI', 'month': 'BM', 'quarter': 'BQ-DEC', 'year': 'BA-DEC'} # Interpretating interval parameter
    bus_index = pd.date_range(range_begin_date, range_end_date, freq = bus_freq_dict[interval]) # Extracting business dates
    
    return bus_index

def hh_drop_nyse_closures(date_index):
# Version 0.01
# Cleaning nyse closure day from the business dates list in form of pd.DatetimeIndex 
# date_index - business dates list in form of pd.DatetimeIndex

# ATTENTION!!! 
# You need to perform the next line to make Jupyter Lab interface to use function:
# pip install pandas_market_calendars

    import pandas as pd # For Timeseries
    from datetime import date # For date/string converting functionality    
    import pandas_market_calendars as mcal # For NYSE Closures
    
    date_format = '%Y-%m-%d'
    
    first_date = date_index[0].strftime(date_format) # Extracting border dates from date_index list
    last_date = date_index[date_index.size - 1].strftime(date_format)
    
    nyse_calendar = mcal.get_calendar('NYSE') # Creating NYSE workdays list
    nyse_valid_days = nyse_calendar.valid_days(first_date, last_date)
    
    ser_temp = pd.Series(0, index = date_index) # Creating temp timeseries for filtering
    date_nyse_index = ser_temp[nyse_valid_days].index # Performing filtering

    return date_nyse_index