# THIS LIBRARY CONTAINS SPECIFIC FUNCTIONS, CREATED FOR PURPOSES OF MARKET RISK INDICATOR PROJECT

def hh_build_mri_from_model(source_file_path, source_model_sheet, hdf_file_path, hdf_object_key, date_index, update_hdf = True):
    """
    Version 0.03
    
    FUNCTIONALITY: 
      TO FILL Functionality description
    OUTPUT:
      TO FILL Returns
    INPUT:
      source_file_path (string) - path to the source data xlsx file
      source_model_sheet (string) - name of the model sheet in source data file sheets list
      hdf_file_path (string) - path to the converted source data keeping HDF5 file
      hdf_object_key (string) - data object key to access converted data from HDF5 file
      date_index (pd.DatetimeIndex) - dates list to take from source file
      update_hdf (boolean) - decision flag if to update data keeping file from the source file or to use existing data keeping file
    """
    
    import numpy as np
    import pandas as pd
    
    # Expanding visibility zone for Python engine to make HH Modules seen
    import sys 
    sys.path.append('../..')
    
    # Including custom functions
    from HH_Modules.hh_files import hh_aggregate_xlsx_tabs
    from HH_Modules.hh_ts import hh_missing_data_manager
    
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
    
    if (update_hdf): # ATTENTION! This part can be eliminated if hdf file with actual data is already stored properly
        # Aggregating data from the source xlsx file to pd.DataFrame    
        ser_tab_list = df_model_asset['Asset Tab Name']
        ser_code_list = df_model_asset['Asset Code']
        df_source_data = hh_aggregate_xlsx_tabs(source_file_path, ser_tab_list, ser_code_list)
        print('hh_build_mri_from_model: Aggregated data table successfully constructed') 

        # Saving aggregated data to HDF5 format with indexation
        df_source_data.to_hdf(hdf_file_path, key = hdf_object_key, mode = 'w', format = 'table', append = False)
        print('hh_build_mri_from_model: HDF5 file', hdf_file_path, 'successfully updated (object key:', hdf_object_key, ')')
    else:
        print('hh_build_mri_from_model: HDF5 file taken as is because of update refusing')

    # Extracting data from HDF file
    date_format = '%Y-%m-%d'    
    first_date = date_index[0].strftime(date_format) 
    last_date = date_index[date_index.size - 1].strftime(date_format)
    df_selected_data = pd.read_hdf(hdf_file_path, hdf_object_key, where = ['index >= first_date & index <= last_date'])
    print('hh_build_mri_from_model: Limited data from HDF5 file', hdf_file_path, 'extracted successfully')  
    
    # Filling missing data with previous filled values for all columns of data table
    df_selected_data = hh_missing_data_manager(df_selected_data, manage_option = 'previous')
    print('hh_build_mri_from_model: Missed data in limited data table filled successfully')
        
    # Completing data table with starting and finishing ranges of rows in case of extrracted date index is shorter than needed
    df_selected_data = df_selected_data.reindex(date_index.tz_localize(None), method = 'ffill')
    print('hh_build_mri_from_model: Missed border date rows in limited data table added')
    
    return df_selected_data # Temporary return to check function work