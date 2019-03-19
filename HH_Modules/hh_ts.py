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


def hh_rolling_percentile(ser_to_manage, min_wnd, max_wnd, min_interpretation = 'not_NaN', manage_option = 'mean'):
    """
    Version 0.02 2019-03-19
    
    FUNCTIONALITY: 
      Convert data vector to vector of percentile ranks of every element in the part of vector, formed as rolling window that ends with this element
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
    """

    import numpy as np
    import pandas as pd
 
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
            rolling_wnd_rank = lambda arr_rolling_wnd: (pd.Series(arr_rolling_wnd).rank(method = 'min').iloc[-1] - 1) / pd.Series(arr_rolling_wnd).notna().sum()
            
        if (manage_option == 'less_equal'):
            rolling_wnd_rank = lambda arr_rolling_wnd: pd.Series(arr_rolling_wnd).rank(pct = True, method = 'max').iloc[-1]
            
        if (manage_option == 'mean'):
            rolling_wnd_rank = lambda arr_rolling_wnd: ((pd.Series(arr_rolling_wnd).rank(method = 'min').iloc[-1] - 1) / pd.Series(arr_rolling_wnd).notna().sum() + pd.Series(arr_rolling_wnd).rank(pct = True, method = 'max').iloc[-1]) / 2             
        # Calculating percentiles
        
        ser_ranks = ser_to_manage.rolling(window = max_wnd, min_periods = min_wnd, win_type = None).apply(rolling_wnd_rank, raw = True)
        
    print('hh_rolling_percentile: Percentile rank calculation with min_interpretation', min_interpretation ,'and option', manage_option ,'performed successfully')
    return ser_ranks


def hh_rolling_simple_MA(ser_to_manage, min_wnd, max_wnd, min_interpretation = 'not_NaN', factor_period = 'year'):
    """
    Version 0.02 2019-03-19
        
    FUNCTIONALITY: 
      Convert data vector to vector of simple moving average means of every element in the part of vector, 
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
    """

    import numpy as np
    import pandas as pd
    from math import sqrt

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
        
    print('hh_rolling_simple_MA: Moving average calculation with min_interpretation', min_interpretation ,'performed successfully')
    return ser_SMA


def hh_rolling_z_score(ser_to_manage, min_wnd, max_wnd, winsor_perc_bottom = 0, winsor_perc_top = 100, manage_option = 'standart'):
    """
    Version 0.01 2019-03-19
    
    FUNCTIONALITY: 
      TO FILL !!! Convert data vector to vector of simple moving avarage means of every element in the part of vector, 
    OUTPUT:
      TO FILL !!! ser_SMA (pd.Series) - processed data vector of simple moving averages
    INPUT:
      ser_to_manage (pd.Series) - source data vector
      min_wnd (integer) - minimal rolling window width
      max_wnd (integer) - maximal rolling window width      
      winsor_perc_bottom (integer) - bottom percentile to set minimal outliers
      winsor_perc_top (integer) - top percentile to set maximal outliers      
      TO FILL!!! manage_option (string) - z filling defining rule: 
    """

    import numpy as np
    import pandas as pd
    
    # Initializing resulting variable
    df_z_score_res = pd.DataFrame()

    # Calculating rolling mean
    ser_rolling_mean = ser_to_manage.rolling(window = max_wnd, min_periods = min_wnd, win_type = None).mean()
    # Calculating rolling standard deviation    
    ser_rolling_std = ser_to_manage.rolling(window = max_wnd, min_periods = min_wnd, win_type = None).std()
    # Calculating rolling z-score
    ser_rolling_z_score = (ser_to_manage - ser_rolling_mean) / ser_rolling_std
    
    df_z_score_res['Mean'] = ser_rolling_mean
    df_z_score_res['Std'] = ser_rolling_std    
    df_z_score_res['Z-Score'] = ser_rolling_z_score     
  
    print('hh_rolling_z_score: Something performed successfully')
    return df_z_score_res