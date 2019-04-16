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