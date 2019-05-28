### THIS LIBRARY CONTAINS GENERAL PURPOSES FILES IMPORTING, EXPORTING AND CONVERTING FUNCTIONS

def hh_aggregate_xlsx_tabs(source_file_path, tab_list, code_list):
    """
    Version 0.02 2019-03-11
    
    FUNCTIONALITY: 
      Imports table data from selected group of tabs of xlsx file to common DataFrame
    OUTPUT:
      df_xlsx_data (pd.DataFrame) - aggregated data table
        'Date' - dates column
        [Asset Code] - columns with index data from particular tab_list item named like corresponding code_list items
    INPUT:
      source_file_path (string) - path to the source data file
      tab_list (pd.Series) - list of tabs to export from files
      code_list (pd.Series) - list of asset codes to name return columns (corresponded with tab_list)
    """
    
    import pandas as pd    

    for counter, (tab_name, asset_code) in enumerate(zip(tab_list, code_list)):
        df_next_tab = pd.read_excel(source_file_path, sheet_name = tab_name, header = 0, index_col = 0, names = [asset_code])
        if (counter == 0):
            df_xslx_data = df_next_tab
        else:
            df_xslx_data = df_xslx_data.join(df_next_tab, how = 'outer')
            
        print('hh_aggregate_xlsx_tabs: Tab', tab_name, '(', asset_code, ') successfully loaded from', source_file_path)        

    print('hh_aggregate_xlsx_tabs: MS EXCEL file', source_file_path, 'successfully exported to aggregated DataFrame')          
    return df_xslx_data


def hh_get_xlsx_risk_events(source_file_path, tab_name):
    """
    Version 0.01 2019-04-16
    
    FUNCTIONALITY: 
      Imports risk events data from xlsx file to DataFrame
    OUTPUT:
      df_risk_events (pd.DataFrame) - data table
    INPUT:
      source_file_path (string) - path to the source data file
      tab_name (string) - tab to export from file
    """    

    import pandas as pd
    from datetime import datetime
    
    events_date_format = '%Y%m%d'
    df_risk_events = pd.read_excel(source_file_path, sheet_name = tab_name, header = 0, index_col = 0)
    df_risk_events['Begin date'] = pd.to_datetime(df_risk_events['Beginning'], format = events_date_format)
    df_risk_events['End date'] = pd.to_datetime(df_risk_events['End'], format = events_date_format)
    df_risk_events.drop(columns = ['Beginning', 'End'], inplace = True)

    print('hh_get_xlsx_risk_events: Risk events data from', source_file_path, 'successfully exported to DataFrame')      
    return df_risk_events


def hh_get_country_codes():
    """
    Version 0.02 2019-04-22
    
    FUNCTIONALITY: 
      Imports and converts country code table from url
    OUTPUT:
      df_result (pd.DataFrame) - data table
    INPUT:
    """ 
    
    import pandas as pd
    
    url_country_code = 'https://countrycode.org/'
    df_full_codes = pd.read_html(url_country_code, index_col = 'COUNTRY')[0]
    df_full_codes[['ISO SHORT', 'ISO LONG']] = df_full_codes['ISO CODES'].str.split(' / ', expand = True)
    df_result = df_full_codes[['ISO SHORT', 'ISO LONG']]      
    df_result.index = df_result.index.str.upper()

    print('hh_get_country_codes: Country codes from', url_country_code, 'successfully exported to DataFrame') 
    return df_result


def hh_get_msci_index_membership(source_file_path):
    """
    Version 0.03 2019-04-22
    
    ATTINTION: Follow the instruction of file preparing!!!
    
    FUNCTIONALITY: 
      Imports and converts MSCI index membership table from file
    OUTPUT:
      df_index_class (pd.DataFrame) - list of indexes and their market classes
      df_index_member (pd.DataFrame) - list of members for each index      
    INPUT:
      source_file_path (string) - path to the source data file
    """
    
    import pandas as pd
    
    df_members = pd.read_excel(source_file_path, sheet_name = 'Indexes', header = 0, index_col = [0, 1])
    df_members = df_members[~ df_members['Member'].str.endswith('Countries')]
    df_members.reset_index(level = 'Market Type', inplace = True)
    df_index_class = df_members['Market Type'].to_frame()
    df_index_class = df_index_class[~ df_index_class.index.duplicated()]     
    df_index_member = df_members['Member'].str.slice(0, -4).to_frame()
    df_index_member['Member Code'] = df_members['Member'].str.slice(-3, -1)
    
    print('hh_get_msci_index_membership: Information about index membership from', source_file_path, 'successfully exported to DataFrame')     
    return [df_index_class, df_index_member]


def hh_get_msci_reclassification(source_file_path):
    """
    Version 0.03 2019-04-24
    
    ATTINTION: Follow the instruction of file preparing!!!
    
    FUNCTIONALITY: 
      Imports and converts MSCI countries reclassification table from file
    OUTPUT:
      df_reclass (pd.DataFrame) - list of reclassifications with dates and directions
    INPUT:
      source_file_path (string) - path to the source data file
    """
    
    import pandas as pd
    
    df_reclass_source = pd.read_excel(source_file_path, sheet_name = 'Reclassifications', header = 2)
    df_reclass_source.head()

    df_reclass_source = df_reclass_source[df_reclass_source['MARKET RECLASSIFICATION'].str.contains(' to ')]
    df_reclass_source['Country'] = df_reclass_source['COUNTRY INDEXES'].str.replace('MSCI ', '').str.replace(' Index', '').str.replace('*', '').str.rstrip().str.upper()
    df_reclass_source[['From Class', 'To Class']] = df_reclass_source['MARKET RECLASSIFICATION'].str.strip(' ').str.extract('(From )(.+?)(.+to )(.)')[[1, 3]] + 'M'
    df_reclass_source['Change Date'] = pd.to_datetime(df_reclass_source['DATE*'], format = '%B %Y') + pd.DateOffset(months = 1) - pd.DateOffset(days = 1)
    df_reclass = df_reclass_source.loc[ : , 'Country' : 'Change Date']
    df_reclass['Country'] = df_reclass['Country'].str.replace(r'&', 'AND')
    
    print('hh_get_msci_reclassification: Information about MSCI reclassifications from', source_file_path, 'successfully exported to DataFrame')    
    return df_reclass    


def hh_get_msci_returns(source_dir_path, str_part_to_replace, dict_to_rename_countries):
    """
    Version 0.02 2019-04-23
    
    ATTENTION: Follow the instruction of files preparing!!!
    
    FUNCTIONALITY: 
      Imports and consolidates MSCI returns info from all suitable files in current directory
    OUTPUT:
      df_returns (pd.DataFrame) - set of returns by date and country/index
    INPUT:
      source_dir_path (string) - path to the directory with source data files
      str_part_to_replace (string) - part of column names to clear country name (usinng as str_part_to_replace + '.+')  
      dict_to_rename_countries (string) - dictionary of MSCI country names changing for compatibility with universal country codes
    """
    
    import pandas as pd
    import os

    date_format = '%Y-%m-%d'
    arr_from_file = []
    dir_MSCI_returns_byte = os.fsencode(source_dir_path)
    for history_file_byte in os.listdir(dir_MSCI_returns_byte):
        history_file_str = os.fsdecode(history_file_byte)
        if history_file_str.startswith('historyIndex') and history_file_str.endswith('.xls'): 
            df_from_file = pd.read_excel(source_dir_path + history_file_str, skiprows = 6, header = 0, dtype={'Date': str})
            df_from_file = df_from_file[ : df_from_file['Date'].isnull().idxmax()]        
            df_from_file['Date'] = pd.to_datetime(df_from_file['Date'], format = date_format)
            df_from_file.set_index('Date', drop = True, inplace = True)
            df_from_file.columns = df_from_file.columns.str.replace(str_part_to_replace + '.+', '').str.rstrip()
            arr_from_file.append(df_from_file)
    df_returns = pd.concat(arr_from_file, axis = 1, join = 'outer')
    df_returns.columns.name = 'Country'
    df_returns.rename({'KOREA': 'SOUTH KOREA', 'USA': 'UNITED STATES'}, axis = 1, inplace  = True)
    
    print('hh_get_msci_returns: Information about MSCI returns from', source_dir_path, 'successfully consolidated to DataFrame')    
    return df_returns


def hh_save_msci_returns(df_to_save, returns_freq, returns_size, returns_style, returns_suite, returns_level, returns_currency, result_file_path):
    """
    Version 0.01 2019-04-23
       
    FUNCTIONALITY: 
      Saves msci returns to hdf file through appending with key formed as "returns_freq/returns_size/returns_style/returns_suite/returns_level/returns_currency"
    OUTPUT:
      object_msci_returns_hdf (string) - key to created/appended object inside the HDF5 file
    INPUT:
      df_to_save (pd.DataFrame) - data set to be stored
      returns_freq (string) - frequency of returns ('daily' / 'monthly' / 'yearly')
      returns_size (string) - size of capitalization for businesses to be included to index ('standart' or other sample)
      returns_style (string) - returns style ('value' / 'growth' / 'none')
      returns_suite (string) - specific way of index calulation ('none' or particular suite)
      returns_level (string) - level of returns index ('price' / 'gross' / 'net')
      returns_currency (string) - currency of returns ('USD' / 'local')      
      result_file_path (string) - path to the hdf file      
    """
    
    import pandas as pd
    import sys 
    sys.path.append('../..')
    
    from HH_Modules.hh_files import hh_get_country_codes
    
    df_country_codes = hh_get_country_codes()
    
    df_to_save_stack = df_to_save.stack().to_frame()
    df_to_save_stack.columns = ['Returns']
    df_to_save_stack = df_to_save_stack.merge(df_country_codes, how = 'left', left_on = 'Country', right_index = True)
    df_to_save_stack.drop('ISO LONG', axis = 1, inplace = True)
    df_to_save_stack.columns = ['Returns', 'Code']
    df_to_save_stack['Code'].fillna('INDEX', inplace = True)
    df_to_save_stack.set_index('Code', append = True, inplace = True)
    
    object_msci_returns_hdf = 'msci_returns_data/' + returns_freq + '/' + returns_size + '/' + returns_style + '/' + returns_suite + '/' + returns_level + '/' + returns_currency
    df_to_save_stack.to_hdf(result_file_path, key = object_msci_returns_hdf, mode = 'a', format = 'table', append = True)

    print('hh_save_msci_returns: Information about MSCI returns with options', object_msci_returns_hdf, 'successfully saved to', result_file_path)    
    return object_msci_returns_hdf


def hh_get_msci_membership_evolution(returns_file_path, returns_key, membership_file_path, reclass_file_path):
    """
    Version 0.02 2019-04-25
       
    FUNCTIONALITY: 
      Forming MSCI history on base of current membership info, reclassifications info and returns start dates    
    OUTPUT:
      df_country_status (pd.DataFrame) - data set of full MSCI history
    INPUT:
      returns_file_path (string) - path to HDF5 file containing returns info to get boundary dates for countries returns
      returns_key (string) - key to object inside the HDF5 file
      membership_file_path (string) - path to xlsx data file with current MSCI membership info
      reclass_file_path (string) - path to xlsx data file with current MSCI reclassifications history
    """
    
    import pandas as pd
    import sys 
    sys.path.append('../..')
    
    from HH_Modules.hh_files import hh_get_msci_index_membership
    from HH_Modules.hh_files import hh_get_msci_reclassification

    date_format = '%Y-%m-%d'
    
    df_returns = pd.read_hdf(returns_file_path, returns_key)
    [df_index_class, df_index_member] = hh_get_msci_index_membership(membership_file_path)
    df_reclass = hh_get_msci_reclassification(reclass_file_path)

    ### Preparing DataFrame with current market indexes info: Country Code(index), Country Name, boundary observations datesand current index membership info:
    df_country_status = pd.DataFrame(index = df_returns.sort_index(level = 'Code').index.unique(level = 'Code'), columns = ['Country', 'Start Date', 'End Date'])
    for country_code in df_country_status.index:
        if (country_code != 'INDEX'):
            df_country_status.loc[country_code, 'Country'] = df_returns.xs(country_code, level = 'Code').index.values[0][1]
            df_country_status.loc[country_code, 'Start Date'] = df_returns.xs(country_code, level = 'Code').index.values[0][0]
            df_country_status.loc[country_code, 'End Date'] = df_returns.xs(country_code, level = 'Code').index.values[-1][0]
        
    df_country_status.drop('INDEX', inplace = True)
    df_country_status = df_country_status.merge(df_index_member.loc[['DM', 'EM', 'FM']].reset_index(), 
                                                how = 'left', left_index = True, right_on = 'Member Code').set_index('Member Code').drop('Member', axis = 1)
    df_country_status.fillna('SM', inplace = True)
    df_country_status.head()
    df_country_status.reset_index(inplace = True)

    ### Integrating reclassifications history to current market indexes info: correcting starting boundary for current membership due to history and adding new history rows:
    df_reclass.sort_values('Change Date', axis = 0, ascending = False, inplace = True)
    for row_index, reclass_row in df_reclass.iterrows():
        df_country_reclass = df_country_status[df_country_status['Country'] == reclass_row['Country']].copy()
        df_country_reclass = df_country_reclass[df_country_reclass['Index Name'] == reclass_row['To Class']]
        if (len(df_country_reclass.index) > 0):
            index_status_row = df_country_reclass['End Date'].idxmin()
            ser_new_country_status = df_country_status.loc[index_status_row].copy()
            ser_new_country_status['End Date'] = reclass_row['Change Date'] - pd.DateOffset(days = 1)
            ser_new_country_status['Index Name'] = reclass_row['From Class']
            df_country_status.loc[index_status_row, 'Start Date'] = reclass_row['Change Date']
            df_country_status = df_country_status.append(ser_new_country_status, ignore_index = True)
        else:
            print('hh_get_msci_classification_evolution: Country not found in current indexes DM, EM, FM and Standalone:', reclass_row['Country'])
            
    df_country_status.sort_values(['Member Code', 'Start Date'], axis = 0, inplace = True)
    df_country_status.set_index('Member Code', drop = True, inplace = True)
    
    print('hh_get_msci_classification_evolution: MSCI membership history successfully formed')
    return df_country_status


def hh_get_ison_universe(source_file_path, df_msci_membership, flag_drop_to_msci = True):
    """
    Version 0.01 2019-04-26
       
    FUNCTIONALITY: 
      Exporting and formatting ISON universe history
    OUTPUT:
      df_country_status (pd.DataFrame) - data set of full MSCI history
    INPUT:
      returns_file_path (string) - path to HDF5 file containing returns info to get boundary dates for countries returns
      returns_key (string) - key to object inside the HDF5 file
      membership_file_path (string) - path to xlsx data file with current MSCI membership info
      reclass_file_path (string) - path to xlsx data file with current MSCI reclassifications history
    """
    
    import pandas as pd
    import sys 
    sys.path.append('../..')
    from datetime import datetime    
    from HH_Modules.hh_files import hh_get_country_codes
    
    df_country_codes = hh_get_country_codes()
    
    df_ison_membership = pd.read_excel(source_file_path)
    df_ison_membership.columns = ['Class Number', 'Country Code', 'Start Date', 'End Date', 'Status', 'Index Name']
    df_ison_membership.drop(['Class Number', 'Status'], axis = 1, inplace = True)
    df_ison_membership['End Date'].fillna(datetime(2018, 12, 31), inplace = True)
    df_ison_membership.set_index('Country Code', inplace = True)
    df_ison_membership.sort_values(['Country Code', 'Start Date'], axis = 0, inplace = True)
    df_ison_membership.loc[df_ison_membership['Start Date'] == '1970-01-01', 'Start Date'] = datetime(1969, 12, 31)
    df_ison_membership = df_ison_membership.merge(df_country_codes.reset_index(), how = 'left', left_index = True, right_on = 'ISO SHORT')
    df_ison_membership.drop('ISO LONG', axis = 1, inplace = True)
    df_ison_membership.columns = ['Start Date', 'End Date', 'Index Name', 'Country', 'Member Code']
    df_ison_membership.set_index('Member Code', inplace = True)
    if (flag_drop_to_msci):
        df_ison_membership = df_ison_membership[df_ison_membership.index.isin(df_msci_membership.index)]
        df_ison_membership = df_ison_membership.reset_index().merge(df_msci_membership['Start Date'].reset_index().groupby('Member Code').min(), how = 'left',
                                                                    left_on = 'Member Code', right_index = True).set_index('Member Code')   
        df_ison_membership.loc[(df_ison_membership['Start Date_x'] < df_ison_membership['Start Date_y']),
                               'Start Date_x'] = df_ison_membership.loc[(df_ison_membership['Start Date_x'] < df_ison_membership['Start Date_y']), 'Start Date_y']
        df_ison_membership.drop('Start Date_y', axis = 1, inplace = True)
        df_ison_membership.rename(columns={'Start Date_x' : 'Start Date'}, inplace = True)
        df_ison_membership.reset_index(inplace = True)
        df_ison_membership.drop(df_ison_membership[df_ison_membership['Start Date'] > df_ison_membership['End Date']].index, inplace = True)
        df_ison_membership.set_index('Member Code', inplace = True)
        
    print('hh_get_ison_universe: Information about ISON Universe successfully successfully exported to DataFrame')    
    return df_ison_membership