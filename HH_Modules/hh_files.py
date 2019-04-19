# THIS LIBRARY CONTAINS GENERAL PURPOSES FILES IMPORTING, EXPORTING AND CONVERTING FUNCTIONS

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
    df_risk_events    
    
    return df_risk_events


def hh_get_country_codes():
    """
    Version 0.01 2019-04-18
    
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
    
    return df_result


def hh_get_msci_index_membership(source_file_path):
    """
    Version 0.02 2019-04-19
    
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
    df_index_member = df_members['Member'].str.slice(0, -4).str.capitalize().to_frame()
    df_index_member['Member Code'] = df_members['Member'].str.slice(-3, -1)
    
    return [df_index_class, df_index_member]


def hh_get_msci_reclassification(source_file_path):
    """
    Version 0.01 2019-04-19
    
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
    df_reclass_source['Country'] = df_reclass_source['COUNTRY INDEXES'].str.replace('MSCI ', '').str.replace(' Index', '').str.replace('*', '').str.rstrip()
    df_reclass_source[['From Class', 'To Class']] = df_reclass_source['MARKET RECLASSIFICATION'].str.strip(' ').str.extract('(From )(.+?)(.+to )(.)')[[1, 3]] + 'M'
    df_reclass_source['Change Date'] = pd.to_datetime(df_reclass_source['DATE*'], format = '%B %Y') + pd.DateOffset(months = 1)
    df_reclass = df_reclass_source.loc[ : , 'Country' : 'Change Date']
    
    return df_reclass    


def hh_get_msci_returns(source_dir_path):
    """
    Version 0.01 2019-04-19
    
    ATTINTION: Follow the instruction of files preparing!!!
    
    FUNCTIONALITY: 
      Imports and consloidates MSCI returns info from all suitable files in current directory
    OUTPUT:
      df_returns (pd.DataFrame) - set of returns by date and country/index
    INPUT:
      source_dir_path (string) - path to the directory with source data files
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
            df_from_file.columns = df_from_file.columns.str.extract('(.+?)( .+)')[0]
            arr_from_file.append(df_from_file)
    df_returns = pd.concat(arr_from_file, axis = 1, join = 'outer')
    
    return df_returns