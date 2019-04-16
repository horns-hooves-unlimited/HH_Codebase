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
    df_model_raw = pd.read_excel(source_file_path, sheet_name = source_model_sheet, header = 1, usecols = [0, 1, 2, 3, 4, 5, 6])
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


def hh_standartize_mri_data(df_model_asset, df_selected_data, date_to_start, MRI_min_wnd, MRI_max_wnd, MRI_winsor_bottom, MRI_winsor_top, hdf_file_path):
    """
    Version 0.12 2019-04-03
    
    FUNCTIONALITY:
      1) Selecting base asset for each asset group and rearranging dataset for base elements priority    
      2) Performing winsorized z-scoring for each asset of each group
      3) For non-base assets: standartizing z-matrix for assets, which have are too many missed values before date_to_start
      4) Forming standartized data vector for each asset from standartized z-matrix or base z-matrix
      5) For all assets: adding weighted z-matrix to group level collection
      6) Calculating group mean matrix for weighted z-matrices in group level collection
      7) Building group z-matrix from group mean matrix
      8) Saving group z-matrix to HDF5 file with reseted index because of HDF5 limitations (index is in 0 column) for further calculations
      9) Calculating percentile data vector for each group z-matrix
    OUTPUT:
      df_asset_standartized (pd.DataFrame) - collection of standartized z-scores (pd.Timeseries) for all assets
      df_group_mean_z_diag (pd.DataFrame) - collection of diagonales of group's z matrices for all groups    
      df_group_percentiled (pd.DataFrame) - collection of percentiled group's z matrices (pd.Timeseries) for all groups 
    INPUT:
      df_model_asset (pd.DataFrame) - asset list and weights descripted at model
      df_selected_data (pd.DataFrame) - main work dataset - result of data import from source file and next transormations
      date_to_start (string) - date for analysis of data availability at staring periods (boundary of starting period)
      MRI_min_wnd (integer) - minimal rolling window width for normalizing function
      MRI_max_wnd (integer) - maximal rolling window width for normalizing function    
      MRI_winsor_bottom (integer) - bottom winsorization boundary for normalizing function
      MRI_winsor_top (integer) - top winsorization boundary for normalizing function
      hdf_file_path (string) - path to save HDF5 file with z matrices for each group      
    """
    
    import numpy as np
    import pandas as pd
    from datetime import datetime
    from HH_Modules.hh_ts import hh_rolling_z_score

    # Initialising function visibility variables
    date_format = '%Y-%m-%d'        
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
            if (df_source[ : date_to_start][asset_code].notna().sum() > int_base_counter):                
                # Determination of base element for group and it's attributes fixation
                int_base_counter = df_source[ : date_to_start][asset_code].notna().sum()                
                base_asset_number = df_model_asset.index.get_loc(asset_index)
                base_asset_code = asset_code
        # Changing assets order within the group for base element priority
        if (first_asset_number != base_asset_number):
            base_oriented_index = df_model_asset.index.tolist()
            base_oriented_index[first_asset_number], base_oriented_index[base_asset_number] = base_oriented_index[base_asset_number], base_oriented_index[first_asset_number]
            df_model_asset = df_model_asset.reindex(base_oriented_index)
            df_model_asset.reset_index(drop = True, inplace = True)
        print('hh_standartize_mri_data: basic asset for group', asset_group_name, 'determined succesfully:', base_asset_code)
        
    # Standartizing cycle on group level
    # Initialising loop visibility variables            
    arr_group_diag_container = []
    arr_group_vector_container = []
    arr_asset_vector_container = []
    arr_asset_codes_global = []
    arr_group_codes = []
    for asset_group_name, df_asset_group in df_model_asset.groupby('Asset Group'):
        # Initialising group visibility variables        
        print('hh_standartize_mri_data: group', asset_group_name, 'standartizing started')
        bool_base_asset = True
        arr_asset_matrix_container = []
        arr_asset_codes = []
        # Standartizing cycle on asset level with the group
        for (asset_index, asset_code) in df_asset_group['Asset Code'].iteritems():
            # Assignment of base asset data set
            if (bool_base_asset):
                bool_base_asset = False
                # Performing z scoring for base asset
                [df_base_z_score, df_base_z_matrix] = hh_rolling_z_score(np.log(df_source[asset_code]), min_wnd = MRI_min_wnd, max_wnd = MRI_max_wnd, winsor_option = 'value', 
                                                                         winsor_bottom = MRI_winsor_bottom, winsor_top = MRI_winsor_top, fill_option = 'backfill')
                ser_base_z_score_winsor = df_base_z_score['Z Winsorized']
                # Calculating etalon filled quantity
                int_base_filled = ser_base_z_score_winsor[ : date_to_start].notna().sum()                
                # Defining of standartized values of base asset as diagonal of z matrix (without backfilling)
                arr_asset_vector_container.append(pd.Series(np.copy(np.diag(df_base_z_matrix)), index = df_base_z_matrix.index))
                # Initialising dataset with non np.NaN wages sum for group
                df_group_weights = pd.DataFrame(np.zeros(df_base_z_matrix.shape), index = df_base_z_matrix.index, columns = df_base_z_matrix.columns)
                # Creating a whole group dataset with multiplying asset matrix to asset weight
                arr_asset_matrix_container.append(df_base_z_matrix * df_model_asset.at[asset_index, 'Factor Weights'])    
                df_group_weights = df_group_weights + df_base_z_matrix.notna() * df_model_asset.at[asset_index, 'Factor Weights']               
                arr_asset_codes.append(asset_code)
                arr_asset_codes_global.append(asset_code)
            # Normalization of other asset's data sets                
            else:
                # Performing z scoring for asset                
                [df_asset_z_score, df_asset_z_matrix] = hh_rolling_z_score(np.log(df_source[asset_code]), min_wnd = MRI_min_wnd, max_wnd = MRI_max_wnd, winsor_option = 'value', 
                                                                           winsor_bottom = MRI_winsor_bottom, winsor_top = MRI_winsor_top, fill_option = 'backfill')
                ser_asset_z_score_simple = df_asset_z_score['Z Score']
                ser_asset_z_score_winsor = df_asset_z_score['Z Winsorized']               
                # Calculating asset filled quantity                
                int_asset_filled = ser_asset_z_score_winsor[ : date_to_start].notna().sum()                
                # Standartizing asset if they do not have enough initial values
                if (int_asset_filled < int_base_filled * double_base_allowed_part):
                    df_asset_start_index = ser_asset_z_score_simple.index.get_loc(ser_asset_z_score_simple.first_valid_index())                 
                    # Renormatizing asset z matrix with base z matrix data
                    for end_wnd_index in range(df_asset_start_index, min(df_asset_start_index + MRI_max_wnd, ser_asset_z_score_simple.size)):
                        ser_base_z_matrix_part = df_base_z_matrix.iloc[max(0, df_asset_start_index - MRI_min_wnd + 1) : end_wnd_index + 1, end_wnd_index]
                        df_asset_z_matrix.iloc[:, end_wnd_index] = df_asset_z_matrix.iloc[:, end_wnd_index] * ser_base_z_matrix_part.std()  + ser_base_z_matrix_part.mean()
                        
                # Defining of standartized values of asset as diagonal of modified z matrix (without backfilling)
                arr_asset_vector_container.append(pd.Series(np.copy(np.diag(df_asset_z_matrix)), index = df_asset_z_matrix.index))            
                # Adding asset matrix to a whole group dataset with multiplying asset matrix to asset weight          
                arr_asset_matrix_container.append(df_asset_z_matrix * df_model_asset.at[asset_index, 'Factor Weights'])  
                df_group_weights = df_group_weights + df_asset_z_matrix.notna() * df_model_asset.at[asset_index, 'Factor Weights']                    
                arr_asset_codes.append(asset_code)   
                arr_asset_codes_global.append(asset_code)                
            print('hh_standartize_mri_data: asset', asset_code, 'in group', asset_group_name, 'standartized successfully') 
        
        # Calculating z matrix for group from weighted asset matrices
        df_group_mean = pd.concat(arr_asset_matrix_container, axis = 0, keys = arr_asset_codes, names = ['Asset Code', 'Date'], copy = False)   
        df_group_mean = df_group_mean.sum(level = 1)
        df_group_mean[df_group_weights > 0] =  df_group_mean[df_group_weights > 0] / df_group_weights[df_group_weights > 0]
        df_group_mean[df_group_weights == 0] = np.NaN
        print('hh_standartize_mri_data: weighted mean matrix for group' , asset_group_name, 'builded successfully')         
        df_group_mean_z = (df_group_mean - df_group_mean.mean())/df_group_mean.std()
        # Adding diagonale of group weighted mean z-score matrix to MRI dataset
        arr_group_diag_container.append(pd.Series(np.copy(np.diag(df_group_mean_z)), index = df_group_mean_z.index))
        print('hh_standartize_mri_data: z-score matrix for group' , asset_group_name, 'weighted mean matrix builded successfully') 
        # Saving group matrix to hdf file for further manipulations
        df_group_to_save = df_group_mean_z.copy()
        df_group_to_save = df_group_to_save.astype(float)
        df_group_to_save.reset_index(inplace = True)
        df_group_to_save.columns = np.arange(len(df_group_to_save.columns))
        df_group_to_save.to_hdf(hdf_file_path, key = asset_group_name, mode = 'a', format = 'fixed') 
        print('hh_standartize_mri_data: z-score matrix for group' , asset_group_name, 'saved to HDF5 file', hdf_file_path, '(object key:', asset_group_name, ')') 
                
        arr_group_codes.append(asset_group_name)
        # Creating data vector of percentiled group's z matrix columns for each group
        ser_group_z_percentile = pd.Series(np.NaN, index = df_group_mean_z.index) 
        ser_group_z_percentile.name = asset_group_name
        for column_index in df_group_mean_z.columns:
            if (column_index >= datetime.strptime(date_to_start, date_format)):
                ser_rolling_wnd = df_group_mean_z.loc[(column_index - pd.DateOffset(years = 1) + pd.DateOffset(days = 1)) : column_index, column_index]
                ser_group_z_percentile[column_index] = ((ser_rolling_wnd.rank(method = 'min')[-1] - 1) / ser_rolling_wnd.notna().sum() + 
                        ser_rolling_wnd.rank(pct = True, method = 'max')[-1]) / 2                    
        arr_group_vector_container.append(ser_group_z_percentile)
        print('hh_standartize_mri_data: percentiled data vector on base of mean z score matrix for group' , asset_group_name, 'builded successfully')                 

    # Collection of standartized z-scores for all assets
    df_asset_standartized = pd.concat(arr_asset_vector_container, axis = 0, keys = arr_asset_codes_global, names = ['Asset Code', 'Date'], copy = False)
    print('hh_standartize_mri_data: asset standartized z-score collection builded successfully')
    # Collection of diagonales of group's z matrices for all groups
    df_group_mean_z_diag = pd.concat(arr_group_diag_container, axis = 0, keys = arr_group_codes, names = ['Group Name', 'Date'], copy = False)
    print('hh_standartize_mri_data: data vector collection of diagonales of mean z score matrix for all groups builded successfully')       
    # Collection of percentiled group's z matrices for all groups
    df_group_percentiled = pd.concat(arr_group_vector_container, axis = 0, keys = arr_group_codes, names = ['Group Name', 'Date'], copy = False)
    print('hh_standartize_mri_data: percentiled data vector collection on base of mean z score matrix for all groups builded successfully')         
    
    return [df_asset_standartized, df_group_mean_z_diag, df_group_percentiled]


def hh_aggregate_mri_data(df_model_MRI, hdf_z_matrix_path, hdf_group_info_path, object_perc_grouped_hdf, ma_max_wnd):
    """
    Version 0.01 2019-04-03
    
    FUNCTIONALITY:
      1) For all groups: extcrating from file and adding weighted z-matrix to MRI level collection
      2) Calculating MRI weighted mean matrix for weighted z-matrices in MRI level collection
      3) Building MRI z-matrix from MRI mean matrix
      4) Building MRI percentiled vector as weighted mean of group percentiled vectors
      9) Calculating moving average vector for MRI percentiled vector
    OUTPUT:
      ser_MRI_mean_z_diag (pd.Series) - diagonale of weighted mean z-score matrix builded from z-score group matrices
      ser_MRI_perc (pd.Series) - weighted mean of percentiled z-matrices for groups   
      ser_MRI_perc_MA (pd.Series) - result of moving average for weighted mean of percentiled z-matrices for groups
    INPUT:
      df_model_MRI (pd.DataFrame) - group list and weights descripted at model
      hdf_z_matrix_path (string) - path to load HDF5 file with z matrices for each group
      hdf_group_info_path (string) - path to load HDF5 file with percentiled z matrices for each group    
      object_perc_grouped_hdf (string) - key for pd.DataFrame object with percentiled z matrices in hdf_group_info_path file
      ma_max_wnd (integer) - maximum windows size for moving aveage calculating from weighted data vector from percentiled group matrices 
    """
    
    import numpy as np
    import pandas as pd
    from datetime import datetime
    from HH_Modules.hh_ts import hh_rolling_simple_MA    

    # Initialising containers for weighted mean matrix calculation
    arr_matrix_container = []
    arr_group_codes = []
    # Group aggregating cycle    
    for group_index, ser_group_info in df_model_MRI.iterrows():
        group_code = ser_group_info['Asset Code']
        group_weight = ser_group_info['Factor Weights']
        # Loading group z score matrix from HDF5 file
        df_group_z_matrix = pd.read_hdf(hdf_z_matrix_path, group_code)
        df_group_z_matrix.set_index(0, drop = True, inplace = True)
        # Initialising not np.NaN weights sum DataFrame
        if (group_index == 0):
            df_group_weights = pd.DataFrame(np.zeros(df_group_z_matrix.shape), index = df_group_z_matrix.index, columns = df_group_z_matrix.columns)
        # Adding weighted matrix to container
        arr_matrix_container.append(df_group_z_matrix * group_weight)
        arr_group_codes.append(group_code)
        # Adding not np.NaN weights to aggregated DataFrame
        df_group_weights = df_group_weights + df_group_z_matrix.notna() * group_weight
        print('hh_aggregate_mri_data: group', group_code, 'z matrix data extracted successfully')

    # Calculating z matrix for MRI from weighted group matrices        
    df_MRI_mean = pd.concat(arr_matrix_container, axis = 0, keys = arr_group_codes, names = ['Group Code', 'Date'], copy = False) 
    df_MRI_mean = df_MRI_mean.sum(level = 1)
    df_group_weights[df_group_weights == 0] = np.NaN
    df_MRI_mean = df_MRI_mean / df_group_weights
    print('hh_aggregate_mri_data: weighted mean matrix for MRI builded successfully')    
    df_MRI_mean_z = (df_MRI_mean - df_MRI_mean.mean())/df_MRI_mean.std()
    ser_MRI_mean_z_diag = pd.Series(np.copy(np.diag(df_MRI_mean_z)), index = df_MRI_mean_z.index)
    print('hh_aggregate_mri_data: z-score matrix for MRI weighted mean matrix builded successfully')     

    # Initialising containers for weighted percentiled vector calculation
    arr_perc_container = []    
    # Group aggregating cycle    
    for group_index_perc, ser_group_info_perc in df_model_MRI.iterrows():
        group_code = ser_group_info_perc['Asset Code']
        group_weight = ser_group_info_perc['Factor Weights']
        # Loading group z scored matrix percentalization result from HDF5 file
        ser_group_perc= pd.read_hdf(hdf_group_info_path, object_perc_grouped_hdf)[group_code]
        # Initialising not np.NaN weights sum TimeSeries
        if (group_index_perc == 0):
            ser_group_weights = pd.Series(np.zeros(len(ser_group_perc)), index = ser_group_perc.index)        
        # Adding weighted matrix to container
        arr_perc_container.append(ser_group_perc * group_weight)        
        # Adding not np.NaN weights to aggregated TimeSeries
        ser_group_weights = ser_group_weights + ser_group_perc.notna() * group_weight
        print('hh_aggregate_mri_data: group', group_code, 'percentiled data vector extracted successfully')
        
    # Calculating z matrix for MRI from weighted group matrices        
    ser_MRI_perc = pd.concat(arr_perc_container, axis = 0, keys = arr_group_codes, names = ['Group Code', 'Date'], copy = False)
    ser_MRI_perc = ser_MRI_perc.sum(level = 1)
    ser_group_weights[ser_group_weights == 0] = np.NaN
    ser_MRI_perc = ser_MRI_perc / ser_group_weights   
    ser_MRI_perc.index = ser_MRI_mean_z_diag.index
    ser_MRI_perc_MA = hh_rolling_simple_MA(ser_MRI_perc, round(ma_max_wnd / 2), ma_max_wnd, show_report = False)
    print('hh_aggregate_mri_data: weighted data vector from percentiled group matrices for MRI and moving average for this vector builded successfully')    
           
    return [ser_MRI_mean_z_diag, ser_MRI_perc, ser_MRI_perc_MA]


def hh_matplotlib_mri_data(df_model_asset, df_model_MRI, asset_hdf_path, group_hdf_path, MRI_hdf_path, asset_selected_key, asset_z_score_key, group_diag_key, group_perc_key, mri_diag_key, mri_raw_key, mri_ma_key, figure_size, figure_hspace, min_date, max_date):
    """
    Version 0.02 2019-04-15
    
    FUNCTIONALITY:
      1) Drawing plot for all groups
      2) Drawing plot for MRI
    OUTPUT:
      dict_fig_MRI (dictionary) - named array, containing group figures and MRI figure
    INPUT:
      df_model_asset (pd.DataFrame) - asset list and weights descripted at model    
      df_model_MRI (pd.DataFrame) - group list and weights descripted at model    
      asset_hdf_path (string) - path to hdf file with asset level data
      group_hdf_path (string) - path to hdf file with group level data
      MRI_hdf_path (string) - path to hdf file with MRI level data     
      asset_selected_key (string) - object key to selected mungled raw source data
      asset_z_score_key (string) - object key to standartized winsorized z-scored selected mungled raw source data
      group_diag_key (string) - object key to diagonales of group z-matrices   
      group_perc_key (string) - object key to percentiled vectors of group z-matrices
      mri_diag_key (string) - object key to diagonale of group MRI z-matrix
      mri_raw_key (string) - object key to percentiled vector of group MRI z-matrix
      mri_ma_key (string) - object key to moving average of percentiled vector of group MRI z-matrix
      figure_size (tuple) - figure shapes
      figure_hspace (float) - height space between plots within the figure
      min_date (datetime) - start date for plotting
      max_date (datetime) - end date for plotting    
    """
    
    import numpy as np
    import pandas as pd
    from datetime import datetime 
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    # Extracting asset level data
    df_raw_data = pd.read_hdf(asset_hdf_path, asset_selected_key)
    df_raw_data.set_index('Date', drop = True, inplace = True)
    df_standart_data = pd.read_hdf(asset_hdf_path, asset_z_score_key)
    df_standart_data.set_index('Date', drop = True, inplace = True)
    # Extracting group level data
    df_group_diag = pd.read_hdf(group_hdf_path, group_diag_key)
    df_group_diag.set_index('Date', drop = True, inplace = True)
    df_group_perc = pd.read_hdf(group_hdf_path, group_perc_key)
    df_group_perc.set_index('Date', drop = True, inplace = True)
    # Extracting MRI level data
    ser_mri_diag = pd.read_hdf(MRI_hdf_path, mri_diag_key)
    ser_mri_raw_perc = pd.read_hdf(MRI_hdf_path, mri_raw_key)
    ser_mri_ma_perc = pd.read_hdf(MRI_hdf_path, mri_ma_key)

    dict_fig_MRI = {}
    # Drawing subplots for each group
    for group_counter, (asset_group_name, df_asset_group) in enumerate(df_model_asset.groupby('Asset Group')):
        # Initialising group visibility variables
        fig_group, axes_group = plt.subplots(3, 1, figsize = figure_size)
        plt.subplots_adjust(hspace = figure_hspace)
        axes_group[0].set_title(asset_group_name + ' / raw series')
        axes_group[1].set_title(asset_group_name + ' / standartized winsorized z-scores of series')  
        axes_group[2].set_title(asset_group_name + ' / group weighted mean')        
        for (asset_index, asset_code) in df_asset_group['Asset Code'].iteritems():    
            # Drawing raw series for current group assets            
            df_raw_data[asset_code].plot(ax = axes_group[0], label = asset_code, alpha = 0.75, grid = True, rot = 0, linestyle = ':')
            # Drawing z-score series for current group assets                       
            df_standart_data[asset_code].plot(ax = axes_group[1], label = asset_code, alpha = 0.75, grid = True, rot = 0, linestyle = ':')
        # Drawing z-score for group weighted mean        
        df_group_diag[asset_group_name].plot(ax = axes_group[2], label = asset_group_name, alpha = 1.0, grid = True, rot = 0) 
        # Configuring plots
        for ax_group in axes_group:
            ax_group.legend(loc = 'best')
            ax_group.xaxis.set_major_locator(mdates.YearLocator())
            ax_group.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax_group.xaxis.set_minor_locator(mdates.MonthLocator()) 
            ax_group.set_xlim(min_date, max_date)             
        # Adding plot to dictionary array        
        dict_fig_MRI[asset_group_name] = fig_group
    
    # Drawing subplots for MRI
    fig_MRI, axes_MRI = plt.subplots(3, 1, figsize = figure_size)    
    plt.subplots_adjust(hspace = figure_hspace)    
    axes_MRI[0].set_title('Separate groups and common MRI z-scores')    
    axes_MRI[1].set_title('Separate groups and common MRI percentiled values')  
    axes_MRI[2].set_title('MRI percentiled values: raw and by moving average')     
    for group_index, ser_group_info in df_model_MRI.iterrows():
        group_code = ser_group_info['Asset Code']
        # Drawing z-score series for current group assets
        df_group_diag[group_code].plot(ax = axes_MRI[0], label = group_code, alpha = 0.75, grid = True, rot = 0, linestyle = ':')        
        # Drawing percentiled series for current group assets        
        df_group_perc[group_code].plot(ax = axes_MRI[1], label = group_code, alpha = 0.75, grid = True, rot = 0, linestyle = ':')
    # Drawing z-score series for MRI
    ser_mri_diag.plot(ax = axes_MRI[0], label = 'MRI', alpha = 1.0, grid = True, rot = 0)    
    # Drawing percentiled series for MRI
    ser_mri_raw_perc.plot(ax = axes_MRI[1], label = 'MRI', alpha = 1.0, grid = True, rot = 0)    
    # Drawing percentiled series for MRI
    ser_mri_raw_perc.plot(ax = axes_MRI[2], label = 'Raw', alpha = 1.0, grid = True)
    ser_mri_ma_perc.plot(ax = axes_MRI[2], label = 'Moving average', alpha = 1.0, grid = True, rot = 0)
    # Configuring plots    
    axes_MRI[0].legend(loc = 'best')    
    axes_MRI[1].legend(loc = 'best')        
    axes_MRI[2].legend(loc = 'best')  
    for ax_MRI in axes_MRI:
        ax_MRI.legend(loc = 'best')
        ax_MRI.xaxis.set_major_locator(mdates.YearLocator())
        ax_MRI.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax_MRI.xaxis.set_minor_locator(mdates.MonthLocator()) 
        ax_MRI.set_xlim(min_date, max_date)
    # Adding plot to dictionary array    
    dict_fig_MRI['MRI'] = fig_MRI
        
    return dict_fig_MRI


def hh_bokeh_mri_data(df_model_asset, df_model_MRI, asset_hdf_path, group_hdf_path, MRI_hdf_path, asset_selected_key, asset_z_score_key, group_diag_key, group_perc_key, mri_diag_key, mri_raw_key, mri_ma_key, figure_size, bokeh_toolbar, min_date, max_date, df_risk_band):
    """
    Version 0.03 2019-04-16
    
    FUNCTIONALITY:
      1) Drawing plot for all groups
      2) Drawing plot for MRI
    OUTPUT:
      arr_figures (array) - array of group figures and MRI figure
    INPUT:
      df_model_asset (pd.DataFrame) - asset list and weights descripted at model    
      df_model_MRI (pd.DataFrame) - group list and weights descripted at model    
      asset_hdf_path (string) - path to hdf file with asset level data
      group_hdf_path (string) - path to hdf file with group level data
      MRI_hdf_path (string) - path to hdf file with MRI level data     
      asset_selected_key (string) - object key to selected mungled raw source data
      asset_z_score_key (string) - object key to standartized winsorized z-scored selected mungled raw source data
      group_diag_key (string) - object key to diagonales of group z-matrices   
      group_perc_key (string) - object key to percentiled vectors of group z-matrices
      mri_diag_key (string) - object key to diagonale of group MRI z-matrix
      mri_raw_key (string) - object key to percentiled vector of group MRI z-matrix
      mri_ma_key (string) - object key to moving average of percentiled vector of group MRI z-matrix
      figure_size (tuple) - figure shapes
      bokeh_toolbar (string) - enumeration of tools for figures
      min_date (datetime) - start date for plotting
      max_date (datetime) - end date for plotting    
      df_risk_band (pd.DataFrame) - list of events to draw bands at plots       
    """
    
    import numpy as np
    import pandas as pd
    from datetime import datetime 
    from bokeh.layouts import column as b_col
    import bokeh.models as b_md    
    from bokeh.palettes import Set2, Set3
    import bokeh.plotting as b_pl
    
    # Defining output to notebook
    b_pl.output_notebook()
    
    # Extracting asset level data
    df_raw_data = pd.read_hdf(asset_hdf_path, asset_selected_key)
    df_raw_data.set_index('Date', drop = True, inplace = True)
    df_standart_data = pd.read_hdf(asset_hdf_path, asset_z_score_key)
    df_standart_data.set_index('Date', drop = True, inplace = True)
    # Extracting group level data
    df_group_diag = pd.read_hdf(group_hdf_path, group_diag_key)
    df_group_diag.set_index('Date', drop = True, inplace = True)
    df_group_perc = pd.read_hdf(group_hdf_path, group_perc_key)
    df_group_perc.set_index('Date', drop = True, inplace = True)
    # Extracting MRI level data
    ser_mri_diag = pd.read_hdf(MRI_hdf_path, mri_diag_key)
    ser_mri_raw_perc = pd.read_hdf(MRI_hdf_path, mri_raw_key)
    ser_mri_ma_perc = pd.read_hdf(MRI_hdf_path, mri_ma_key)

    # Initialising of resulting array
    arr_figures = []
    # Initialising of x-axis array for all plots
    arr_index_values = df_raw_data.index.values

    # Drawing plots for asset groups
    for group_counter, (asset_group_name, df_asset_group) in enumerate(df_model_asset.groupby('Asset Group')):
        # Defining color palette for each group
        num_asset_lines = len(df_asset_group['Asset Code'])
        palette_asset_lines = Set2[8][ : num_asset_lines]
        # Initialising raw asset plot for group
        fig_group_raw = b_pl.figure(tools = bokeh_toolbar, x_axis_label = 'Date', plot_width = figure_size[0], plot_height = figure_size[1], x_axis_type = 'datetime', 
                                    title = asset_group_name + ' / raw series', x_range = (min_date, max_date))
        # Initialising z-score asset plot for group        
        fig_group_z_score = b_pl.figure(tools = bokeh_toolbar, x_axis_label = 'Date', plot_width = figure_size[0], plot_height = figure_size[1], x_axis_type = 'datetime',
                                        title = asset_group_name + ' / standartized winsorized z-scores of series', x_range = fig_group_raw.x_range)        
        # Drawing asset lines for asset plots
        for asset_counter, asset_code in enumerate(df_asset_group['Asset Code']):
            fig_group_raw.line(x = arr_index_values, y = df_raw_data[asset_code].values,
                               line_color = palette_asset_lines[asset_counter], line_width = 1, legend = asset_code, name = str(asset_counter))   
            fig_group_z_score.line(x = arr_index_values, y = df_standart_data[asset_code].values,
                               line_color = palette_asset_lines[asset_counter], line_width = 1, legend = asset_code, name = str(asset_counter)) 
        # Initialising and drawing consolidated line for group z-score
        fig_group_mean = b_pl.figure(tools = bokeh_toolbar, x_axis_label = 'Date', plot_width = figure_size[0], plot_height = figure_size[1], x_axis_type = 'datetime',
                                     title = asset_group_name + ' / group weighted mean', x_range = fig_group_raw.x_range)
        fig_group_mean.line(x = arr_index_values, y = df_group_diag[asset_group_name].values, legend = asset_group_name)

        # Tuning plots and adding risk event bands
        fig_group_raw.legend.location = 'top_left'
        fig_group_raw.legend.background_fill_alpha  = 0.75
        fig_group_raw.legend.spacing  = 0        
        fig_group_raw.xaxis.axis_label_standoff = 0
        fig_group_raw.title.align = 'center'   
        fig_group_raw.min_border_bottom = 50
        for event_counter, ser_event in df_risk_band.iterrows():
            fig_group_raw.add_layout(b_md.BoxAnnotation(left = ser_event['Begin date'], right = ser_event['End date'], fill_alpha=0.1, fill_color='red'))
        fig_group_raw.add_tools(b_md.HoverTool(tooltips = [('Date', '@x{%F}')], formatters = {'x': 'datetime'}, mode = 'vline', names = ['0']))            
        fig_group_z_score.legend.location = 'top_left'                
        fig_group_z_score.legend.background_fill_alpha  = 0.75        
        fig_group_z_score.legend.spacing  = 0   
        fig_group_z_score.xaxis.axis_label_standoff = 0        
        fig_group_z_score.title.align = 'center'    
        fig_group_z_score.min_border_bottom = 50
        for event_counter, ser_event in df_risk_band.iterrows():
            fig_group_z_score.add_layout(b_md.BoxAnnotation(left = ser_event['Begin date'], right = ser_event['End date'], fill_alpha=0.1, fill_color='red'))        
        fig_group_z_score.add_tools(b_md.HoverTool(tooltips = [('Date', '@x{%F}')], formatters = {'x': 'datetime'}, mode = 'vline', names = ['0']))             
        fig_group_mean.legend.location = 'top_left'
        fig_group_mean.legend.background_fill_alpha  = 0.75  
        fig_group_mean.legend.spacing  = 0        
        fig_group_mean.xaxis.axis_label_standoff = 0
        fig_group_mean.title.align = 'center'
        fig_group_mean.min_border_bottom = 50        
        for event_counter, ser_event in df_risk_band.iterrows():
            fig_group_mean.add_layout(b_md.BoxAnnotation(left = ser_event['Begin date'], right = ser_event['End date'], fill_alpha=0.1, fill_color='red'))
        fig_group_mean.add_tools(b_md.HoverTool(tooltips = [('Date', '@x{%F}')], formatters = {'x': 'datetime'}, mode = 'vline'))
        # Consloidating plots for asset group and adding layout to resulting plot
        arr_figures.append(b_col(fig_group_raw, fig_group_z_score, fig_group_mean))

    # Defining color palette for MRI
    num_group_lines = len(df_model_MRI['Asset Code'])
    palette_group_lines = Set2[8][ : num_group_lines]  
    # Initialising z-score and percentiled values plots for MRI
    fig_MRI_z_score = b_pl.figure(tools = bokeh_toolbar, x_axis_label = 'Date', plot_width = figure_size[0], plot_height = figure_size[1], x_axis_type = 'datetime',
                                  title = 'Separate groups and common MRI z-scores', x_range = (min_date, max_date))  
    fig_MRI_perc = b_pl.figure(tools = bokeh_toolbar, x_axis_label = 'Date', plot_width = figure_size[0], plot_height = figure_size[1], x_axis_type = 'datetime',
                               title = 'Separate groups and common MRI percentiled values', x_range = fig_MRI_z_score.x_range)  
    # Drawing lines for groups
    for group_counter, group_code in enumerate(df_model_MRI['Asset Code']):
            fig_MRI_z_score.line(x = arr_index_values, y = df_group_diag[group_code].values, 
                                 line_color = palette_group_lines[group_counter], line_width = 1, legend = group_code) 
            fig_MRI_perc.line(x = arr_index_values, y = df_group_perc[group_code].values,
                              line_color = palette_group_lines[group_counter], line_width = 1, legend = group_code, name = str(group_counter))             
    # Drawing line for MRI
    fig_MRI_z_score.line(x = arr_index_values, y = ser_mri_diag.values, legend = 'MRI z score', line_width = 2, name = '0')
    # Drawing line for MRI percentiled
    fig_MRI_perc.line(x = arr_index_values, y = ser_mri_raw_perc.values, legend = 'MRI percentile', color = 'blue')    
    # Initialising plot for MRI percentiled
    fig_MRI_res = b_pl.figure(tools = bokeh_toolbar, x_axis_label = 'Date', plot_width = figure_size[0], plot_height = figure_size[1], x_axis_type = 'datetime',
                              title = 'MRI percentiled values: raw and by moving average', x_range = fig_MRI_z_score.x_range)      
    # Drawing lines for MRI percentiled (raw and moving average)
    fig_MRI_res.line(x = arr_index_values, y = ser_mri_raw_perc.values, legend = 'Raw', color = 'blue', name = '0')   
    fig_MRI_res.line(x = arr_index_values, y = ser_mri_ma_perc.values, legend = 'MA-5', color = 'orange') 
    # Tuning plots and adding risk event bands
    fig_MRI_z_score.legend.location = 'top_left'
    fig_MRI_z_score.legend.background_fill_alpha  = 0.75    
    fig_MRI_z_score.legend.spacing  = 0        
    fig_MRI_z_score.xaxis.axis_label_standoff = 0
    fig_MRI_z_score.title.align = 'center'  
    fig_MRI_z_score.min_border_bottom = 50 
    for event_counter, ser_event in df_risk_band.iterrows():
        fig_MRI_z_score.add_layout(b_md.BoxAnnotation(left = ser_event['Begin date'], right = ser_event['End date'], fill_alpha=0.1, fill_color='red'))    
    fig_MRI_z_score.add_tools(b_md.HoverTool(tooltips = [('Date', '@x{%F}')], formatters = {'x': 'datetime'}, mode = 'vline', names = ['0']))          
    fig_MRI_perc.legend.location = 'top_left'
    fig_MRI_perc.legend.background_fill_alpha  = 0.75
    fig_MRI_perc.legend.spacing  = 0        
    fig_MRI_perc.xaxis.axis_label_standoff = 0
    fig_MRI_perc.title.align = 'center'
    fig_MRI_perc.min_border_bottom = 50    
    for event_counter, ser_event in df_risk_band.iterrows():
        fig_MRI_perc.add_layout(b_md.BoxAnnotation(left = ser_event['Begin date'], right = ser_event['End date'], fill_alpha=0.1, fill_color='red'))   
    fig_MRI_perc.add_tools(b_md.HoverTool(tooltips = [('Date', '@x{%F}')], formatters = {'x': 'datetime'}, mode = 'vline', names = ['0']))          
    fig_MRI_res.legend.location = 'top_left'
    fig_MRI_res.legend.background_fill_alpha  = 0.75
    fig_MRI_res.legend.spacing  = 0       
    fig_MRI_res.xaxis.axis_label_standoff = 0    
    fig_MRI_res.title.align = 'center'
    fig_MRI_res.min_border_bottom = 50     
    for event_counter, ser_event in df_risk_band.iterrows():
        fig_MRI_res.add_layout(b_md.BoxAnnotation(left = ser_event['Begin date'], right = ser_event['End date'], fill_alpha=0.1, fill_color='red'))     
    fig_MRI_res.add_tools(b_md.HoverTool(tooltips = [('Date', '@x{%F}')], formatters = {'x': 'datetime'}, mode = 'vline', names = ['0']))
    
    # Consloidating plots for asset group and adding layout to resulting plot
    arr_figures.append(b_col(fig_MRI_z_score, fig_MRI_perc, fig_MRI_res))       
        
    return arr_figures