# THIS LIBRARY CONTAINS GENERAL PURPOSES DATES MANIPULATING FUNCTIONS

def hh_create_bus_dates(date_type = 'string', begin_date = '1900-01-01', end_date = '2018-12-31', interval = 'day'):
# Version 0.02
# FUNCTIONALITY: 
#  Creating the business dates list in form of pd.DatetimeIndex
# OUTPUT:
#  bus_index (pd.DatetimeIndex) - business dates list
# INPUT:
#  date_type (string) - type of first and last dates representation: 'string' = '%Y-%m-%d'; 'date' = datetime.date
#  begin_date (datetime.date or string) - first date of dates list in date_type mentioned format
#  end_date (datetime.date or string) - last date of dates list in date_type mentioned format
#  interval (string) - period of dates list: 'day' = days; 'week' = week ends; 'month' = month ends; 'quarter' = quarter ends; 'year' = year ends

    import pandas as pd # For date_range functionality
    from datetime import date # For date/string converting functionality
    
    date_format = '%Y-%m-%d'
    
    # Converting dates to common string format in case of date formatting
    if date_type == 'string': 
        range_begin_date = begin_date
        range_end_date = end_date
    else:
        range_begin_date = begin_date.strftime(date_format)
        range_end_date = end_date.strftime(date_format)        
        
    # Interpretating interval parameter    
    bus_freq_dict = {'day': 'B', 'week': 'W-FRI', 'month': 'BM', 'quarter': 'BQ-DEC', 'year': 'BA-DEC'} 
    # Extracting business dates
    bus_index = pd.date_range(range_begin_date, range_end_date, freq = bus_freq_dict[interval]) 
    
    print('hh_create_bus_dates: Business dates index for period from',  begin_date, 'to', end_date, 'with', interval, 'interval successfully generated')
    return bus_index

def hh_drop_nyse_closures(date_index):
# Version 0.02
# FUNCTIONALITY:
#  Cleaning nyse closure days from the business dates list in form of pd.DatetimeIndex
# OUTPUT:
#  nyse_index (pd.DatetimeIndex object) - cleaned from NYSE closures business dates list
# INPUT:
#  date_index (pd.DatetimeIndex) - business dates list

# ATTENTION!!! 
#  You need to perform the next line to make Jupyter Lab engine to use key function get_calendar:
#  pip install pandas_market_calendars

    import pandas as pd # For Timeseries
    from datetime import date # For date/string converting functionality    
    import pandas_market_calendars as mcal # For NYSE Closures
    
    date_format = '%Y-%m-%d'
    
    # Extracting border dates from date_index list
    first_date = date_index[0].strftime(date_format) 
    last_date = date_index[date_index.size - 1].strftime(date_format)
    
    # Creating NYSE workdays list
    nyse_calendar = mcal.get_calendar('NYSE') 
    nyse_valid_days = nyse_calendar.valid_days(first_date, last_date)
    
    # Creating temp timeseries for filtering
    ser_temp = pd.Series(0, index = date_index) 
    # Performing filtering
    nyse_index = ser_temp[nyse_valid_days].index 
    
    print('hh_drop_nyse_closures: NYSE closure dates successfully dropped from date index')
    return nyse_index