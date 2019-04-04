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
        df_next_tab = pd.read_excel(source_file_path, sheet_name = tab_name, header = 0, usecols = 2, index_col = 0, names = [asset_code])
        if (counter == 0):
            df_xslx_data = df_next_tab
        else:
            df_xslx_data = df_xslx_data.join(df_next_tab, how = 'outer')
            
        print('hh_aggregate_xlsx_tabs: Tab', tab_name, '(', asset_code, ') successfully loaded from', source_file_path)        

    print('hh_aggregate_xlsx_tabs: MS EXCEL file', source_file_path, 'successfully exported to aggregated DataFrame')          
    return df_xslx_data