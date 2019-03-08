# THIS LIBRARY CONTAINS SPECIFIC FUNCTIONS, CREATED FOR PURPOSES OF MARKET RISK INDICATOR PROJECT

def hh_import_mri_source(source_file_path, source_model_sheet, date_index):
# Version 0.01
# TO FILL Functionality description
# TO FILL Returns
# source_file_path (string) - path to the source data file
# source_model_sheet (string) - name of th model sheet in source data file sheets list
# date_index (pd.DatetimeIndex) - dates list to take from source file

    import pandas as pd
    
    # Reading Model information from Source model sheet
    df_model_raw = pd.read_excel(source_file_path, sheet_name = source_model_sheet, header = 1, usecols = 6)
    print('hh_import_mri_source: Model profile successfully read')
    
    # Dropping asset group border rows
    
    
    # Filling empty factor weights with zeros
    df_model_raw['Factor Weights'].fillna(0, inplace = True)
    
    # Negative factor weights detecting
    if (df_model_raw[df_model_raw['Factor Weights'] < 0].size > 0):
        print('hh_import_mri_source: WARNING! Negative factor weights detected')
    
    # Empty asset groups markers detecting        
    if (df_model_raw['Asset Group'].isnull().sum() > 0):
        print('hh_import_mri_source: WARNING! Empty Asset Group marker detected')
    
    # Group border rows deleting
    df_model_raw = df_model_raw[df_model_raw['Asset Group'] != df_model_raw['Asset Code']]
    print('hh_import_mri_source: Group border rows successfully dropped')
    
    # Checking control sums for every group (= 1)
    ser_sum_check = df_model_raw.groupby('Asset Group').sum()
    index_sum_check = ser_sum_check[ser_sum_check['Factor Weights'] < 0.9999].index
    if (index_sum_check.size > 0):
        print('hh_import_mri_source: WARNING! Incorrect group sum weights detected for next groups list:', index_sum_check)
    else:
        print('hh_import_mri_source: Group sum weights control successfully performed')
    
    # Dividing list on asset part and MRI weights part
    df_model_asset = df_model_raw[df_model_raw['Asset Group'] != 'MRI'] # Asset part
    print('hh_import_mri_source: Model asset part extracted')    
    df_model_MRI = df_model_raw[df_model_raw['Asset Group'] == 'MRI'] # MRI part
    print('hh_import_mri_source: Model MRI part extracted')    
    
    return df_model_raw # Temporary return to check function work