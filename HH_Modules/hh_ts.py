# THIS LIBRARY CONTAINS GENERAL PURPOSES WRANGLING AND MUNGLING FUNCTIONS FOR TIMESERIES, BASED ON PANDAS SERIES AND DATAFRAMES
    
def hh_missing_data_manager(df_to_manage, manage_option = 'previous', remove_empty_rows = False, prev_lag = 1000000):
    """
    Version 0.02 2019-03-12
    
    FUNCTIONALITY: 
      Substitutes NaN elements in data table columns with values determined by option defined rule
    OUTPUT:
      df_managed (pd.DataFrame) - processed data table
    INPUT:
      df_to_manage (pd.DataFrame) - source data table
      manage_option (string) - source changing rule: 
        'clear' - removing all data table rows that have at least one NaN value;
        'mean' - substituting by mean value of the particular column;
        'median' - substituting by median value of the particular column;        
        'previous' (default) - substituting by last nearest correct value in the particular column taking into account prev_lag parameter;
      remove_empty_rows (boolean) - wether to remove all row from data table if all column values in it is NaN (default = False) 
      prev_lag - maximum deep of nearest value searching for 'previous' option (default = 1000000)
    """
    
    # Making a copy of data table for further manipulations
    df_managed = df_to_manage.copy()
    
    # Removing full NaN rows
    if (remove_empty_rows):
        df_managed.dropna(how = 'all', inplace = True)
    # Removing rows with at list one NaN value
    if (manage_option == 'clear'):
        df_managed.dropna(how = 'any', inplace = True)
    # Subtituting with means
    if (manage_option == 'mean'):
        df_managed.fillna(value = df_managed.mean(axis = 0), inplace = True)
    # Subtituting with medians
    if (manage_option == 'median'):
        df_managed.fillna(value = df_managed.median(axis = 0), inplace = True)
    # Subtituting with previous
    if (manage_option == 'previous'):
        df_managed.fillna(method = 'ffill', limit = prev_lag, inplace = True)
    
    print('hh_missing_data_manager: np.Nan substitution with option', manage_option, 'performed successfully')  
    print('hh_missing_data_manager: Overall count of actual np.Nan values in data table is', df_managed.isnull().sum().sum())
    
    return df_managed


def hh_rolling_percentile(ser_to_manage, min_wnd, max_wnd, min_interpretation = 'not_NaN', manage_option = 'mean', show_report = False):
    """
    Version 0.04 2019-04-01
    
    FUNCTIONALITY: 
      Converts data vector to vector of percentile ranks of every element in the part of vector, formed as rolling window that ends with this element
    OUTPUT:
      ser_ranks (pd.Series) - processed data vector of percentile ranks
    INPUT:
      ser_to_manage (pd.Series) - source data vector
      min_wnd (integer) - minimal rolling window width
      max_wnd (integer) - maximal rolling window width      
      min_interpretation (string) - minimal quantity defining rule:
          'not_NaN' (default) - the quantity of not NaN elements need to be more or equal min_wnd 
          'any' - the quantity of all elements need to be more or equal min_wnd with at least one not Nan       
      manage_option (string) - rank defining rule: 
        'less' - comparing particular element to compute part of vector elements in window, that are strictly less than particular element;
        'less_equal' - comparing particular element to compute part of vector elements in window, that are less or equals to particular element;
        'mean' (default) - mean of results of 'less' and 'less_equal' manage_option variants applying;        
      show_report (boolean) - flag of showing function resulting report: 
        False (default) - not to show;
        True - to show;        
    """

    import numpy as np
    import pandas as pd
 
    # Checking input parameters
    if (min_wnd <= 0):
        min_wnd = 1
        print('hh_rolling_percentile: WARNING! Min_wnd parameter value is not positive, so it is changed to', min_wnd)
    if (max_wnd < min_wnd):
        max_wnd = min_wnd
        print('hh_rolling_percentile: WARNING! Max_wnd parameter value is less than min_wnd parameter value, so it is changed to', max_wnd)
        
    if (min_interpretation == 'any'):
        # Initializing resulting variable
        ser_ranks = pd.Series(np.NaN, index = ser_to_manage.index)

        for end_wnd_index in range(min_wnd, ser_to_manage.size + 1):
            # Isolating rolling window for particular data vector element
            start_wnd_index = max(0, end_wnd_index - max_wnd)
            ser_rolling_wnd = ser_to_manage[start_wnd_index : end_wnd_index]

            # Checking for not all elements in rolling window are np.NaN
            if (ser_rolling_wnd.notna().sum() == 0):
                continue

            # Calculating percentiles
            if (manage_option == 'less'):
                ser_ranks[end_wnd_index - 1] = (ser_rolling_wnd.rank(method = 'min')[-1] - 1) / ser_rolling_wnd.notna().sum()
                
            if (manage_option == 'less_equal'):
                ser_ranks[end_wnd_index - 1] = ser_rolling_wnd.rank(pct = True, method = 'max')[-1]
                
            if (manage_option == 'mean'):
                ser_ranks[end_wnd_index - 1] = ((ser_rolling_wnd.rank(method = 'min')[-1] - 1) / ser_rolling_wnd.notna().sum() + 
                ser_rolling_wnd.rank(pct = True, method = 'max')[-1]) / 2
    else:
        # Defining calculating function for each manage option
        if (manage_option == 'less'):
            rolling_wnd_rank = lambda ser_rolling_wnd: (ser_rolling_wnd.rank(method = 'min').iloc[-1] - 1) / ser_rolling_wnd.notna().sum()
            
        if (manage_option == 'less_equal'):
            rolling_wnd_rank = lambda ser_rolling_wnd: ser_rolling_wnd.rank(pct = True, method = 'max').iloc[-1]
            
        if (manage_option == 'mean'):
            rolling_wnd_rank = lambda ser_rolling_wnd: ((ser_rolling_wnd.rank(method = 'min').iloc[-1] - 1) / ser_rolling_wnd.notna().sum() + ser_rolling_wnd.rank(pct = True, method = 'max').iloc[-1]) / 2             
        # Calculating percentiles
        
        ser_ranks = ser_to_manage.rolling(window = max_wnd, min_periods = min_wnd, win_type = None).apply(rolling_wnd_rank, raw = False)
    
    if (show_report):
        print('hh_rolling_percentile: Percentile rank calculation with min_interpretation', min_interpretation ,'and option', manage_option ,'performed successfully')
    return ser_ranks


def hh_rolling_simple_MA(ser_to_manage, min_wnd, max_wnd, min_interpretation = 'not_NaN', factor_period = 'year', show_report = False):
    """
    Version 0.05 2019-04-01
        
    FUNCTIONALITY: 
      Converts data vector to vector of simple moving average means of every element in the part of vector, 
      formed as rolling window that ends with this element
    OUTPUT:
      ser_SMA (pd.Series) - processed data vector of simple moving averages
    INPUT:
      ser_to_manage (pd.Series) - source data vector
      min_wnd (integer) - minimal rolling window width
      max_wnd (integer) - maximal rolling window width
      min_interpretation (string) - minimal quantity defining rule:
          'not_NaN' (default) - the quantity of not NaN elements need to be more or equal min_wnd 
          'any' - the quantity of all elements need to be more or equal min_wnd with at least one not Nan          
      factor_period (string) - rank defining rule: 
        'day' = daily; 
        'month' = monthly; 
        'year' (default) = yearly;   
      show_report (boolean) - flag of showing function resulting report: 
        False (default) - not to show;
        True - to show;          
    """

    import numpy as np
    import pandas as pd
    from math import sqrt
 
    # Checking input parameters
    if (min_wnd <= 0):
        min_wnd = 1
        print('hh_rolling_simple_MA: WARNING! Min_wnd parameter value is not positive, so it is changed to', min_wnd)
    if (max_wnd < min_wnd):
        max_wnd = min_wnd
        print('hh_rolling_simple_MA: WARNING! Max_wnd parameter value is less than min_wnd parameter value, so it is changed to', max_wnd)
        
    # Annual factor determining
    annual_factor_dict = {'day': 260, 'month': 12, 'year': 1}
    annual_factor = sqrt(annual_factor_dict[factor_period])
        
    if (min_interpretation == 'any'):
        # Initializing resulting variable
        ser_SMA = pd.Series(np.NaN, index = ser_to_manage.index)

        for end_wnd_index in range(min_wnd, ser_to_manage.size + 1):
            # Isolating rolling window for particular data vector element
            start_wnd_index = max(0, end_wnd_index - max_wnd)
            ser_rolling_wnd = ser_to_manage[start_wnd_index : end_wnd_index]

            # Checking for not all elements in rolling window are np.NaN
            if (ser_rolling_wnd.notna().sum() == 0):
                continue

            # Calculating moving average
            ser_SMA[end_wnd_index - 1] = ser_rolling_wnd.mean() * annual_factor
            
    else: # min_interpretation = 'not_NaN'
        # Calculating moving average
        ser_SMA = ser_to_manage.rolling(window = max_wnd, min_periods = min_wnd, win_type = None).mean() * annual_factor 
    
    if (show_report):
        print('hh_rolling_simple_MA: Moving average calculation with min_interpretation', min_interpretation ,'performed successfully')
    return ser_SMA

def hh_rolling_z_score(ser_to_manage, min_wnd, max_wnd, winsor_option = 'percent', winsor_bottom = 0, winsor_top = 1, fill_option = 'standard', show_report = False):
    """
    Version 0.03 2019-04-01
    
    FUNCTIONALITY: 
      1) Calculates rolling means, deviations and z scores for source data vector
      2) Winsorizing z data vector for each rolling window
      2) Creating z matrix from z winsorized data vectors for each rolling window
    OUTPUT:
      df_z_score_res (pd.DataFrame) - set of data vectors:
            'Mean' (pd.Series) - rolling means with defined window parameters
            'Std' - rolling standard deviations with defined window parameters
            'Z Score' - rolling normalized z scores with defined window parameters
            'Z Winsorized' - rolling normalized z scores with defined window parameters and winsorizing rules
      df_z_matrix (pd.DataFrame) - set of z scores: each columns is a z score vector for the corresponding rolling window with defined window parameters and winsorizing rules
    INPUT:
      ser_to_manage (pd.Series) - source data vector
      min_wnd (integer) - minimal rolling window width (the quantity of not NaN elements in rolling window need to be more or equal min_wnd)
      max_wnd (integer) - maximal rolling window width     
      winsor_option (string) - winsorisation borders interpretating rule:
            'none' - no winsorizing
            'percent' (default) - as percent values from 0 to 1
            'value' - as scalar values without limitations
      winsor_bottom (integer) - bottom boundary of preliminary calculated z-scores to set minimal outliers
      winsor_top (integer) - top boundary of preliminary calculated z-scores to set maximal outliers      
      fill_option (string) - winorized z vector filling defining rule:
             'standard' - only diagonal values of z matrix             
             'backfill' - diagonal values of z matrix added with values of first not NaN column of z matrix
      show_report (boolean) - flag of showing function resulting report: 
        False (default) - not to show;
        True - to show;               
    """

    import numpy as np
    import pandas as pd
    
    # Checking input parameters
    if (min_wnd <= 0):
        min_wnd = 1
        print('hh_rolling_z_score: WARNING! Min_wnd parameter value is not positive, so it is changed to', min_wnd)
    if (max_wnd < min_wnd):
        max_wnd = min_wnd
        print('hh_rolling_z_score: WARNING! Max_wnd parameter value is less than min_wnd parameter value, so it is changed to', max_wnd)
    if (winsor_option == 'percent'):
        if (winsor_bottom < 0):
            winsor_bottom = 0
            print('hh_rolling_z_score: WARNING! Winsor_bottom parameter value is less than 0, so it is changed to', winsor_bottom)
        if (winsor_bottom > 1):
            winsor_bottom = 1
            print('hh_rolling_z_score: WARNING! Winsor_bottom parameter value is more than 1, so it is changed to', winsor_bottom)
        if (winsor_top > 1):
            winsor_top = 1
            print('hh_rolling_z_score: WARNING! Winsor_top parameter value is more than 1, so it is changed to', winsor_top)
        if (winsor_bottom > winsor_top):
            winsor_top = winsor_bottom
            print('hh_rolling_z_score: WARNING! Winsor_top parameter value is less than winsor_bottom parameter value, so it is changed to', winsor_top)
    if (winsor_option == 'value'):
        if (winsor_bottom > winsor_top):
            winsor_top = winsor_bottom
            print('hh_rolling_z_score: WARNING! Winsor_top parameter value is less than winsor_bottom parameter value, so it is changed to', winsor_top)        

    # Initializing resulting variables
    date_format = '%Y-%m-%d'
    df_z_score_res = pd.DataFrame()
    df_z_score_res.index.name = ser_to_manage.name
    df_z_matrix = pd.DataFrame(np.NaN, index = ser_to_manage.index.copy(), columns = ser_to_manage.index.copy())
    df_z_matrix.index.name = ser_to_manage.name
    
    # Calculating rolling mean
    ser_rolling_mean = ser_to_manage.rolling(window = max_wnd, min_periods = min_wnd, win_type = None).mean()
    # Calculating rolling standard deviation    
    ser_rolling_std = ser_to_manage.rolling(window = max_wnd, min_periods = min_wnd, win_type = None).std()
    # Calculating rolling z-score
    ser_rolling_z_score = (ser_to_manage - ser_rolling_mean) / ser_rolling_std
    if (show_report):
        print('hh_rolling_z_score: Mean, Std and Z Score series calculated successfully')

    #Calculating z-score matrix
    for end_wnd_index in range(min_wnd, ser_to_manage.size + 1):        
        # Isolating rolling window for particular data vector element
        start_wnd_index = max(0, end_wnd_index - max_wnd)
        ser_rolling_wnd = ser_to_manage[start_wnd_index : end_wnd_index]

        ser_z_scores = pd.Series(np.NaN, index = ser_to_manage.index)
        # Checking for at list min_wnd elements of rolling window are np.NaN
        if (ser_rolling_wnd.notna().sum() >= min_wnd):
            ser_z_scores = (ser_rolling_wnd - ser_rolling_wnd.mean()) / ser_rolling_wnd.std()
            
            # Winsorization process
            if (winsor_option == 'none'):
                bool_to_winsor = False
            else:
                bool_to_winsor = True
            
            while (bool_to_winsor):       
                # Value based winsorization
                if (winsor_option == 'value'):
                    ser_z_scores.clip(lower = winsor_bottom, upper = winsor_top, inplace = True)
                if (winsor_option == 'percent'):
                    ser_z_scores.clip(lower = ser_z_scores.quantile(winsor_bottom), upper = ser_z_scores.quantile(winsor_top), inplace = True)
                    bool_to_winsor = False
                # Recalculating of z scores
                ser_z_scores = (ser_z_scores - ser_z_scores.mean()) / ser_z_scores.std()                
                # Checking for boundaries
                if (winsor_option == 'value'):
                    if (((ser_z_scores[ser_z_scores < winsor_bottom].size) + (ser_z_scores[ser_z_scores > winsor_top].size)) == 0):
                        bool_to_winsor = False
                        
            # Filling z matrix column part after the winsorizing (if needed)
            df_z_matrix.iloc[start_wnd_index : end_wnd_index, end_wnd_index - 1] = ser_z_scores.values

    if (show_report):
        print('hh_rolling_z_score: Z Matrix values calculated successfully')
    
    # Getting winsorized z meanings       
    ser_z_winsorized = pd.Series(np.copy(np.diag(df_z_matrix)), index = ser_to_manage.index)
    # Backfilling with first not NaN column of z matrix
    if (fill_option == 'backfill'):
        ind_valid_index = ser_z_winsorized.index.get_loc(ser_z_winsorized.first_valid_index())
        ser_z_winsorized[ : ind_valid_index] = df_z_matrix.iloc[ : ind_valid_index, ind_valid_index]
    if (show_report):        
        print('hh_rolling_z_score: Rolling winsorized Z Score series calculated successfully')
    
    df_z_score_res['Mean'] = ser_rolling_mean
    df_z_score_res['Std'] = ser_rolling_std    
    df_z_score_res['Z Score'] = ser_rolling_z_score
    df_z_score_res['Z Winsorized'] = ser_z_winsorized
    
    if (show_report):    
        print('hh_rolling_z_score: Calculating Z Score data with winsor_option', winsor_option, 'and fill_option', fill_option, 'performed successfully')
    return [df_z_score_res, df_z_matrix]


def hh_rolling_mean_subtraction(ser_source, window_size, size_measure, show_report = False):
    """
    Version 0.01 2019-05-03
    
    FUNCTIONALITY:
      Creates a set of data vectors by computing of rolling back window mean subtraction for each source data set element
    OUTPUT:
      ser_mean_subtracted (pd.Series) - resulting data vector for each date point
    INPUT:
      ser_source (pd.Series) - source data vector
      window_size (integer) - size of rolling back window to subtract mean and form data vector
      size_measure (string) - measure of window size: 
        'days' - for days offset;
        'months' - for months offset;
        'years' - for years offset; 
      show_report (boolean) - flag of showing function resulting report: 
        False (default) - not to show;
        True - to show;           
    """
    
    import pandas as pd        
    import datetime
    
    # Creating offset for rolling back window
    offset_day = pd.DateOffset(days = 1)    
    if (size_measure == 'days'):
        offset_window = pd.DateOffset(days = window_size)
    if (size_measure == 'months'):
        offset_window = pd.DateOffset(months = window_size)
    if (size_measure == 'years'):
        offset_window = pd.DateOffset(years = window_size)                
    # Making a copy of data table for further manipulations
    ser_managed = ser_source.copy()
    # Preparing for iterating
    ser_managed.dropna(inplace = True)
    date_ser_min = ser_managed.index[0]
    arr_ser_container = []
    arr_date_container = []
    # Iterating source data vector    
    for date_ser_iter, num_ser_iter in ser_managed.iteritems():
        if (date_ser_iter >= date_ser_min + offset_window):
            # Selecting window to subtract mean and subtracting
            ser_to_subtract = ser_managed[(date_ser_iter - offset_window + offset_day) : date_ser_iter]
            ser_subtracted = ser_to_subtract - ser_to_subtract.mean()
            # Adding new element to data container
            arr_date_container.append(date_ser_iter)            
            arr_ser_container.append(ser_subtracted)
    # Creating resulting data vector
    ser_mean_subtracted = pd.concat(arr_ser_container, axis = 0, keys = arr_date_container, names = ['Value Date', 'Rolling Date'], copy = False)   
    
    if (show_report):
        print('hh_rolling_mean_subtraction: Computing of rolling window mean subtraction performed successfully')    
    return ser_mean_subtracted

