# THIS LIBRARY CONTAINS SPECIFIC FUNCTIONS, CREATED FOR PURPOSES OF MARKET RISK INDICATOR PROJECT

def hh_build_mri_from_model(source_file_path, source_model_sheet, hdf_file_path, hdf_object_key, date_index, update_hdf = True):
    """
    Version 0.04 2019-03-13
    
    FUNCTIONALITY: 
      1) Taking, checking and converting model information from the source xlsx file
      2) Extracting asset and MRI descriptions from model information
      3) Creating/updating HDF5 file with structured source data (if needed)
      4) Taking structured data for selected date interval from HDF5 file
      5) Filling missed data by unlimited forward filling
    OUTPUT:
      df_model_asset (pd.DataFrame) - asset list and weights descripted at model tab
      df_model_MRI (pd.DataFrame) - market risk indexes list and weights descripted at model tab 
      df_selected_data (pd.DataFrame) - result of data import from source file and next transormations
    INPUT:
      source_file_path (string) - path to the source data xlsx file
      source_model_sheet (string) - name of the model sheet in source data file sheets list
      hdf_file_path (string) - path to the converted source data keeping HDF5 file
      hdf_object_key (string) - data object key to access converted data from HDF5 file
      date_index (pd.DatetimeIndex) - dates list to take from source file
      update_hdf (boolean) - decision flag if to update data keeping file from the source file or to use existing data keeping file (default - True)
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
    df_model_asset.reset_index(drop = True, inplace = True)
    print('hh_build_mri_from_model: Model asset part extracted')    
    df_model_MRI = df_model_raw[df_model_raw['Asset Group'] == 'MRI'] # MRI part
    df_model_MRI.reset_index(drop = True, inplace = True)
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
        
    # Completing data table with starting and finishing ranges of rows in case of extrracted date index is shorter than needed
    df_selected_data = df_selected_data.reindex(date_index.tz_localize(None))
    print('hh_build_mri_from_model: Missed border date rows in limited data table added')
    
    # Filling missing data with previous filled values for all columns of data table
    df_selected_data = hh_missing_data_manager(df_selected_data, manage_option = 'previous')
    df_selected_data.index.name = 'Date'
    print('hh_build_mri_from_model: Missed data in limited data table filled successfully')

    
    return [df_model_asset, df_model_MRI, df_selected_data]

def hh_standartize_mri_data(df_model_asset, df_selected_data, date_to_start, MRI_min_wnd, MRI_max_wnd, MRI_winsor_bottom, MRI_winsor_top):
    """
    Version 0.04 2019-03-29
    
    FUNCTIONALITY: TO FILL!!!
      1) Selecting base element for each asset group and rearranging dataset for base elements priority
#      2) Extracting asset and MRI descriptions from model information
#      3) Creating/updating HDF5 file with structured source data (if needed)
#      4) Taking structured data for selected date interval from HDF5 file
#      5) Filling missed data by unlimited forward filling
    OUTPUT: TO FILL!!!
#      df_model_asset (pd.DataFrame) - asset list and weights descripted at model tab
#      df_model_MRI (pd.DataFrame) - market risk indexes list and weights descripted at model tab 
#      df_selected_data (pd.DataFrame) - result of data import from source file and next transormations
    INPUT:
      df_model_asset (pd.DataFrame) - asset list and weights descripted at model
      df_selected_data (pd.DataFrame) - main work dataset - result of data import from source file and next transormations
      date_to_start (string) - date for analysis of data availability at staring periods (boundary of starting period)
      MRI_min_wnd (integer) - minimal rolling window width for normalizing function
      MRI_max_wnd (integer) - maximal rolling window width for normalizing function    
      MRI_winsor_bottom (integer) - bottom winsorization boundary for normalizing function
      MRI_winsor_top (integer) - top winsorization boundary for normalizing function
    """
    
    import numpy as np
    import pandas as pd
    from HH_Modules.hh_ts import hh_rolling_z_score

    # Initialising function visibility variables
    df_source = df_selected_data.copy()
    double_base_allowed_part = 2/3
    
    # Base assets determination  
    for asset_group_name, df_asset_group in df_model_asset.groupby('Asset Group'):
        # Initialising cycle visibility variables
        int_base_counter = 0
        bool_first_asset_in_group = True
        # Determination base asset for group
        for (asset_index, asset_code) in df_asset_group['Asset Code'].iteritems():
            if (bool_first_asset_in_group):
                # First group element's attributes fixation
                first_asset_number = df_model_asset.index.get_loc(asset_index)
                bool_first_asset_in_group = False
            if (df_source.loc[df_source.index < date_to_start][asset_code].notna().sum() > int_base_counter):
                # Determination of base element for group and it's attributes fixation
                int_base_counter = df_source.loc[df_source.index < date_to_start][asset_code].notna().sum()
                base_asset_number = df_model_asset.index.get_loc(asset_index)
        # Changing assets order within the group for base element priority
        if (first_asset_number != base_asset_number):
            base_oriented_index = df_model_asset.index.tolist()
            base_oriented_index[first_asset_number], base_oriented_index[base_asset_number] = base_oriented_index[base_asset_number], base_oriented_index[first_asset_number]
            df_model_asset = df_model_asset.reindex(base_oriented_index)
            df_model_asset.reset_index(drop = True, inplace = True)
            
    # Standartizing cycle on group level
    # Initialising loop visibility variables            
    arr_group_container = []
    arr_group_codes = []
    for asset_group_name, df_asset_group in df_model_asset.groupby('Asset Group'):
        # Initialising geoup visibility variables        
        print('hh_standartize_mri_data: group', asset_group_name, 'started standartizing')
        bool_base_asset = True
        arr_asset_container = []
        arr_asset_codes = []
        # Standartizing cycle on asset level with the group
        for (asset_index, asset_code) in df_asset_group['Asset Code'].iteritems():
            # Assignment of base asset data set
            print('hh_standartize_mri_data: asset', asset_code, 'in group', asset_group_name, 'started standartizing')
            if (bool_base_asset):
                bool_base_asset = False
                # Performing z scoring for base asset
                [df_base_z_score, df_base_z_matrix] = hh_rolling_z_score(np.log(df_source[asset_code]), min_wnd = MRI_min_wnd, max_wnd = MRI_max_wnd, winsor_option = 'value', 
                                                                         winsor_bottom = MRI_winsor_bottom, winsor_top = MRI_winsor_top, fill_option = 'backfill')
                ser_base_z_score_winsor = df_base_z_score['Z Winsorized']
                # Calculating etalon filled quantity
                int_base_filled = ser_base_z_score_winsor[ser_base_z_score_winsor.index < date_to_start].notna().sum()
                # Defining of standartized values of base asset as diagonal of z matrix (without backfilling)
                df_base_z_score['Z Standarized'] = pd.Series(np.copy(np.diag(df_base_z_matrix)), index = df_base_z_matrix.index)  
                # Initialising dataset with non np.NaN wages sum for group
                df_group_weights = pd.DataFrame(np.zeros(df_base_z_matrix.shape), index = df_base_z_matrix.index, columns = df_base_z_matrix.columns)
                # Creating a whole group dataset with multiplying asset matrix to asset weight
                arr_asset_container.append(df_base_z_matrix * df_model_asset.at[asset_index, 'Factor Weights'])    
                df_group_weights = df_group_weights + df_base_z_matrix.notna() * df_model_asset.at[asset_index, 'Factor Weights']               
                arr_asset_codes.append(asset_code)
            # Normalization of other asset's data sets                
            else:
                # Performing z scoring for asset                
                [df_asset_z_score, df_asset_z_matrix] = hh_rolling_z_score(np.log(df_source[asset_code]), min_wnd = MRI_min_wnd, max_wnd = MRI_max_wnd, winsor_option = 'value', 
                                                                           winsor_bottom = MRI_winsor_bottom, winsor_top = MRI_winsor_top, fill_option = 'backfill')
                ser_asset_z_score_simple = df_asset_z_score['Z Score']
                ser_asset_z_score_winsor = df_asset_z_score['Z Winsorized']               
                # Calculating asset filled quantity                
                int_asset_filled = ser_asset_z_score_winsor[ser_asset_z_score_winsor.index < date_to_start].notna().sum()
                # Standartizing asset if they do not have enough initial values
                if (int_asset_filled < int_base_filled * double_base_allowed_part):
                    df_asset_start_index = ser_asset_z_score_simple.index.get_loc(ser_asset_z_score_simple.first_valid_index())                 
                    # Renormatizing asset z matrix with base z matrix data
                    for end_wnd_index in range(df_asset_start_index, min(df_asset_start_index + MRI_max_wnd, ser_asset_z_score_simple.size)):
                        ser_base_z_matrix_part = df_base_z_matrix.iloc[max(0, df_asset_start_index - MRI_min_wnd + 1) : end_wnd_index + 1, end_wnd_index]
                        df_asset_z_matrix.iloc[:, end_wnd_index] = df_asset_z_matrix.iloc[:, end_wnd_index] * ser_base_z_matrix_part.std()  + ser_base_z_matrix_part.mean()
                        
                # Defining of standartized values of asset as diagonal of modified z matrix (without backfilling)
                df_asset_z_score['Z Standarized'] = pd.Series(np.copy(np.diag(df_asset_z_matrix)), index = df_asset_z_matrix.index)
                # Adding asset matrix to a whole group dataset with multiplying asset matrix to asset weight          
                arr_asset_container.append(df_asset_z_matrix * df_model_asset.at[asset_index, 'Factor Weights'])  
                df_group_weights = df_group_weights + df_asset_z_matrix.notna() * df_model_asset.at[asset_index, 'Factor Weights']                    
                arr_asset_codes.append(asset_code)                
            print('hh_standartize_mri_data: asset', asset_code, 'in group', asset_group_name, 'standartized successfully') 
#            if (asset_code == 'iv_us'): # TEMP for testing purposes
#                break
        
        df_group_mean = pd.concat(arr_asset_container, axis = 0, keys = arr_asset_codes, names = ['Asset Code', 'Date'], copy = False)   
        df_group_mean = df_group_mean.sum(level = 1)
        df_group_mean[df_group_weights > 0] =  df_group_mean[df_group_weights > 0] / df_group_weights[df_group_weights > 0]
        df_group_mean[df_group_weights == 0] = np.NaN
        
        if (asset_group_name == 'EQ'): # TEMP for testing purposes
            break
                
    return df_group_mean  # Temporary return