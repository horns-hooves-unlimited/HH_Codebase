def create_bus_dates(begin_date = '1900-01-01', end_date = '2018-12-31', interval = 'day'):
# Version 0.01
# Creating the business dates list in form of pd.DatetimeIndex
# begin_date - first date of dates list
# begin_date - last date of dates list
# interval - period of dates list: 'day' - days, 'week' - week ends, 'month' - month ends, 'quarter' - quarter ends, 'year' - year ends
    import pandas as pd
    
    bus_freq_dict = {'day': 'B', 'week': 'W-FRI', 'month': 'BM', 'quarter': 'BQ-DEC', 'year': 'BA-DEC'}
    bus_index = pd.date_range(begin_date, end_date, freq = bus_freq_dict[interval])
    return bus_index