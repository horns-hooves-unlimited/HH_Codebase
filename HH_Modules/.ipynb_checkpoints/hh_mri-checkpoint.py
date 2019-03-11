# THIS LIBRARY CONTAINS SPECIFIC FUNCTIONS, CREATED FOR PURPOSES OF MARKET RISK INDICATOR PROJECT

def hh_build_mri_from_model(source_file_path, source_model_sheet, date_index):
# Version 0.03
# FUNCTIONALITY: 
#  TO FILL Functionality description
# OUTPUT:
#  TO FILL Returns
# INPUT:
#  source_file_path (string) - path to the source data file
#  source_model_sheet (string) - name of the model sheet in source data file sheets list
#  date_index (pd.DatetimeIndex) - dates list to take from source file

    import pandas as pd
    
    # Expanding visibility zone for Python engine to make HH Modules seen
    import sys 
    sys.path.append('../..')
    
    # Including custom functions
    from HH_Modules.hh_files import hh_aggregate_xlsx_tabs
    
    # Reading Model information from Source model sheet
    df_model_raw = pd.read_excel(source_file_path, sheet_name = source_model_sheet, header = 1, usecols = 6)
    print('hh_build_mri_from_model: Model profile successfully read')
       
    # Filling empty factor weights with zeros
    df_model_raw['Factor Weights'].fillna(0, inplace = True)
    
    # Negative factor weights detecting
    if (df_model_raw[df_model_raw['Factor Weights'] < 0].size > 0):
        print('hh_build_mri_from_model: WARNING! Negative factor weights detected')
    
    # Empty asset groups markers detecting        
    if (df_model_raw['Asset Group'].isnull().sum() > 0):
        print('hh_build_mri_from_model: WARNING! Empty Asset Group marker detected')
    
    # Group border rows deleting
    df_model_raw = df_model_raw[df_model_raw['Asset Group'] != df_model_raw['Asset Code']]
    print('hh_build_mri_from_model: Group border rows successfully dropped')
    
    # Checking control sums for every group (= 1)
    ser_sum_check = df_model_raw.groupby('Asset Group').sum()
    index_sum_check = ser_sum_check[ser_sum_check['Factor Weights'] < 0.9999].index
    if (index_sum_check.size > 0):
        print('hh_build_mri_from_model: WARNING! Incorrect group sum weights detected for next groups list:', index_sum_check)
    else:
        print('hh_build_mri_from_model: Group sum weights control successfully performed')
    
    # Dividing list on asset part and MRI weights part
    df_model_asset = df_model_raw[df_model_raw['Asset Group'] != 'MRI'] # Asset part
    print('hh_build_mri_from_model: Model asset part extracted')    
    df_model_MRI = df_model_raw[df_model_raw['Asset Group'] == 'MRI'] # MRI part
    print('hh_build_mri_from_model: Model MRI part extracted')    
    
    # Aggregating data from the source xlsx file to pd.DataFrame
    ser_tab_list = df_model_asset['Asset Tab Name']
    ser_code_list = df_model_asset['Asset Code']
    df_source_data = hh_aggregate_xlsx_tabs(source_file_path, ser_tab_list, ser_code_list)
    print('hh_build_mri_from_model: Aggregated data table successfully constructed') 
        
    return df_source_data # Temporary return to check function work