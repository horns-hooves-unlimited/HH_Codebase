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
        'previous' - substituting by last nearest correct value in the particular column taking into account prev_lag parameter;
      remove_empty_rows (boolean) - wether to remove all row from data table if all column values in it is NaN  
      prev_lag - maximum deep of nearest value searching for 'previous' option
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
    
