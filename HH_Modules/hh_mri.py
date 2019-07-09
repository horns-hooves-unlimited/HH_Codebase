### THIS LIBRARY CONTAINS SPECIFIC FUNCTIONS, CREATED FOR PURPOSES OF MARKET RISK INDICATOR PROJECT

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
    ### Expanding visibility zone for Python engine to make HH Modules seen:
    import sys 
    sys.path.append('../..')    
    ### Including custom functions:
    from HH_Modules.hh_files import hh_aggregate_xlsx_tabs
    from HH_Modules.hh_ts import hh_missing_data_manager
    
    ### Reading Model information from Source model sheet:
    df_model_raw = pd.read_excel(source_file_path, sheet_name = source_model_sheet, header = 1, usecols = [0, 1, 2, 3, 4, 5, 6])
    print('hh_build_mri_from_model: Model profile successfully read')       
    ### Filling empty factor weights with zeros:
    df_model_raw['Factor Weights'].fillna(0, inplace = True)    
    ### Negative factor weights detecting:
    if (df_model_raw[df_model_raw['Factor Weights'] < 0].size > 0):
        print('hh_build_mri_from_model: WARNING! Negative factor weights detected')    
    ### Empty asset groups markers detecting:       
    if (df_model_raw['Asset Group'].isnull().sum() > 0):
        print('hh_build_mri_from_model: WARNING! Empty Asset Group marker detected')   
    ### Group border rows deleting:
    df_model_raw = df_model_raw[df_model_raw['Asset Group'] != df_model_raw['Asset Code']]
    print('hh_build_mri_from_model: Group border rows successfully dropped')    
    ### Checking control sums for every group (= 1):
    ser_sum_check = df_model_raw.groupby('Asset Group').sum()
    index_sum_check = ser_sum_check[ser_sum_check['Factor Weights'] < 0.9999].index
    if (index_sum_check.size > 0):
        print('hh_build_mri_from_model: WARNING! Incorrect group sum weights detected for next groups list:', index_sum_check)
    else:
        print('hh_build_mri_from_model: Group sum weights control successfully performed')   
    ### Dividing list on asset part and MRI weights part:
    df_model_asset = df_model_raw[df_model_raw['Asset Group'] != 'MRI'] ### Asset part
    df_model_asset.reset_index(drop = True, inplace = True)
    print('hh_build_mri_from_model: Model asset part extracted')    
    df_model_MRI = df_model_raw[df_model_raw['Asset Group'] == 'MRI'] ### MRI part
    df_model_MRI.reset_index(drop = True, inplace = True)
    print('hh_build_mri_from_model: Model MRI part extracted')    
    
    if (update_hdf): ### ATTENTION! This part can be eliminated if hdf file with actual data is already stored properly
        ### Aggregating data from the source xlsx file to pd.DataFrame:
        ser_tab_list = df_model_asset['Asset Tab Name']
        ser_code_list = df_model_asset['Asset Code']
        df_source_data = hh_aggregate_xlsx_tabs(source_file_path, ser_tab_list, ser_code_list)
        print('hh_build_mri_from_model: Aggregated data table successfully constructed') 
        ### Saving aggregated data to HDF5 format with indexation:
        df_source_data.to_hdf(hdf_file_path, key = hdf_object_key, mode = 'w', format = 'table', append = False)
        print('hh_build_mri_from_model: HDF5 file', hdf_file_path, 'successfully updated (object key:', hdf_object_key, ')')
    else:
        print('hh_build_mri_from_model: HDF5 file taken as is because of update refusing')
    ### Extracting data from HDF file:
    date_format = '%Y-%m-%d'    
    first_date = date_index[0].strftime(date_format) 
    last_date = date_index[date_index.size - 1].strftime(date_format)
    df_selected_data = pd.read_hdf(hdf_file_path, hdf_object_key, where = ['index >= first_date & index <= last_date'])
    print('hh_build_mri_from_model: Limited data from HDF5 file', hdf_file_path, 'extracted successfully')          
    ### Completing data table with starting and finishing ranges of rows in case of extrracted date index is shorter than needed:
    df_selected_data = df_selected_data.reindex(date_index.tz_localize(None))
    print('hh_build_mri_from_model: Missed border date rows in limited data table added')    
    ### Filling missing data with previous filled values for all columns of data table:
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

    ### Initialising function visibility variables:
    date_format = '%Y-%m-%d'        
    df_source = df_selected_data.copy()
    double_base_allowed_part = 2/3    
    ### Base assets determination:
    for asset_group_name, df_asset_group in df_model_asset.groupby('Asset Group'):
        ### Initialising cycle visibility variables:
        int_base_counter = 0
        bool_first_asset_in_group = True
        ### Determination base asset for group:
        for (asset_index, asset_code) in df_asset_group['Asset Code'].iteritems():
            if (bool_first_asset_in_group):
                ### First group element's attributes fixation:
                first_asset_number = df_model_asset.index.get_loc(asset_index)
                bool_first_asset_in_group = False
            if (df_source[ : date_to_start][asset_code].notna().sum() > int_base_counter):                
                ### Determination of base element for group and it's attributes fixation:
                int_base_counter = df_source[ : date_to_start][asset_code].notna().sum()                
                base_asset_number = df_model_asset.index.get_loc(asset_index)
                base_asset_code = asset_code
        ### Changing assets order within the group for base element priority:
        if (first_asset_number != base_asset_number):
            base_oriented_index = df_model_asset.index.tolist()
            base_oriented_index[first_asset_number], base_oriented_index[base_asset_number] = base_oriented_index[base_asset_number], base_oriented_index[first_asset_number]
            df_model_asset = df_model_asset.reindex(base_oriented_index)
            df_model_asset.reset_index(drop = True, inplace = True)
        print('hh_standartize_mri_data: basic asset for group', asset_group_name, 'determined succesfully:', base_asset_code)        
    ### Initialising loop visibility variables:          
    arr_group_diag_container = []
    arr_group_vector_container = []
    arr_asset_vector_container = []
    arr_asset_codes_global = []
    arr_group_codes = []
    ### Standartizing loop on group level:
    for asset_group_name, df_asset_group in df_model_asset.groupby('Asset Group'):
        ### Initialising group visibility variables:
        print('hh_standartize_mri_data: group', asset_group_name, 'standartizing started')
        bool_base_asset = True
        arr_asset_matrix_container = []
        arr_asset_codes = []
        ### Standartizing cycle on asset level with the group:
        for (asset_index, asset_code) in df_asset_group['Asset Code'].iteritems():
            ### Assignment of base asset data set:
            if (bool_base_asset):
                bool_base_asset = False
                ### Performing z scoring for base asset:
                [df_base_z_score, df_base_z_matrix] = hh_rolling_z_score(np.log(df_source[asset_code]), min_wnd = MRI_min_wnd, max_wnd = MRI_max_wnd, winsor_option = 'value', 
                                                                         winsor_bottom = MRI_winsor_bottom, winsor_top = MRI_winsor_top, fill_option = 'backfill')
                ser_base_z_score_winsor = df_base_z_score['Z Winsorized']
                ### Calculating etalon filled quantity:
                int_base_filled = ser_base_z_score_winsor[ : date_to_start].notna().sum()                
                ### Defining of standartized values of base asset as diagonal of z matrix (without backfilling):
                arr_asset_vector_container.append(pd.Series(np.copy(np.diag(df_base_z_matrix)), index = df_base_z_matrix.index))
                ### Initialising dataset with non np.NaN wages sum for group:
                df_group_weights = pd.DataFrame(np.zeros(df_base_z_matrix.shape), index = df_base_z_matrix.index, columns = df_base_z_matrix.columns)
                ### Creating a whole group dataset with multiplying asset matrix to asset weight:
                arr_asset_matrix_container.append(df_base_z_matrix * df_model_asset.at[asset_index, 'Factor Weights'])    
                df_group_weights = df_group_weights + df_base_z_matrix.notna() * df_model_asset.at[asset_index, 'Factor Weights']               
                arr_asset_codes.append(asset_code)
                arr_asset_codes_global.append(asset_code)
            ### Normalization of other asset's data sets:                
            else:
                ### Performing z scoring for asset:                
                [df_asset_z_score, df_asset_z_matrix] = hh_rolling_z_score(np.log(df_source[asset_code]), min_wnd = MRI_min_wnd, max_wnd = MRI_max_wnd, winsor_option = 'value', 
                                                                           winsor_bottom = MRI_winsor_bottom, winsor_top = MRI_winsor_top, fill_option = 'backfill')
                ser_asset_z_score_simple = df_asset_z_score['Z Score']
                ser_asset_z_score_winsor = df_asset_z_score['Z Winsorized']               
                ### Calculating asset filled quantity:                
                int_asset_filled = ser_asset_z_score_winsor[ : date_to_start].notna().sum()                
                ### Standartizing asset if they do not have enough initial values:
                if (int_asset_filled < int_base_filled * double_base_allowed_part):
                    df_asset_start_index = ser_asset_z_score_simple.index.get_loc(ser_asset_z_score_simple.first_valid_index())                 
                    ### RenormaLizing asset z matrix with base z matrix data:
                    for end_wnd_index in range(df_asset_start_index, min(df_asset_start_index + MRI_max_wnd, ser_asset_z_score_simple.size)):
                        ser_base_z_matrix_part = df_base_z_matrix.iloc[max(0, df_asset_start_index - MRI_min_wnd + 1) : end_wnd_index + 1, end_wnd_index]
                        df_asset_z_matrix.iloc[:, end_wnd_index] = df_asset_z_matrix.iloc[:, end_wnd_index] * ser_base_z_matrix_part.std()  + ser_base_z_matrix_part.mean()
                       
                ### Defining of standartized values of asset as diagonale of modified z matrix (without backfilling):
                arr_asset_vector_container.append(pd.Series(np.copy(np.diag(df_asset_z_matrix)), index = df_asset_z_matrix.index))            
                ### Adding asset matrix to a whole group dataset with multiplying asset matrix to asset weight:          
                arr_asset_matrix_container.append(df_asset_z_matrix * df_model_asset.at[asset_index, 'Factor Weights'])  
                df_group_weights = df_group_weights + df_asset_z_matrix.notna() * df_model_asset.at[asset_index, 'Factor Weights']                    
                arr_asset_codes.append(asset_code)   
                arr_asset_codes_global.append(asset_code)                
            print('hh_standartize_mri_data: asset', asset_code, 'in group', asset_group_name, 'standartized successfully')         
        ### Calculating z matrix for group from weighted asset matrices:
        df_group_mean = pd.concat(arr_asset_matrix_container, axis = 0, keys = arr_asset_codes, names = ['Asset Code', 'Date'], copy = False)   
        df_group_mean = df_group_mean.sum(level = 1)
        df_group_mean[df_group_weights > 0] =  df_group_mean[df_group_weights > 0] / df_group_weights[df_group_weights > 0]
        df_group_mean[df_group_weights == 0] = np.NaN
        print('hh_standartize_mri_data: weighted mean matrix for group' , asset_group_name, 'builded successfully')         
        df_group_mean_z = (df_group_mean - df_group_mean.mean())/df_group_mean.std()
        ### Adding diagonale of group weighted mean z-score matrix to MRI dataset:
        arr_group_diag_container.append(pd.Series(np.copy(np.diag(df_group_mean_z)), index = df_group_mean_z.index))
        print('hh_standartize_mri_data: z-score matrix for group' , asset_group_name, 'weighted mean matrix builded successfully') 
        ### Saving group matrix to hdf file for further manipulations:
        df_group_to_save = df_group_mean_z.copy()
        df_group_to_save = df_group_to_save.astype(float)
        df_group_to_save.reset_index(inplace = True)
        df_group_to_save.columns = np.arange(len(df_group_to_save.columns))
        df_group_to_save.to_hdf(hdf_file_path, key = asset_group_name, mode = 'a', format = 'fixed') 
        print('hh_standartize_mri_data: z-score matrix for group' , asset_group_name, 'saved to HDF5 file', hdf_file_path, '(object key:', asset_group_name, ')')                 
        arr_group_codes.append(asset_group_name)
        ### Creating data vector of percentiled group's z matrix columns for each group:
        ser_group_z_percentile = pd.Series(np.NaN, index = df_group_mean_z.index) 
        ser_group_z_percentile.name = asset_group_name
        for column_index in df_group_mean_z.columns:
            if (column_index >= datetime.strptime(date_to_start, date_format)):
                ser_rolling_wnd = df_group_mean_z.loc[(column_index - pd.DateOffset(years = 1) + pd.DateOffset(days = 1)) : column_index, column_index]
                ser_group_z_percentile[column_index] = ((ser_rolling_wnd.rank(method = 'min')[-1] - 1) / ser_rolling_wnd.notna().sum() + 
                        ser_rolling_wnd.rank(pct = True, method = 'max')[-1]) / 2                    
        arr_group_vector_container.append(ser_group_z_percentile)
        print('hh_standartize_mri_data: percentiled data vector on base of mean z score matrix for group' , asset_group_name, 'builded successfully')                 
    ### Collection of standartized z-scores for all assets:
    df_asset_standartized = pd.concat(arr_asset_vector_container, axis = 0, keys = arr_asset_codes_global, names = ['Asset Code', 'Date'], copy = False)
    print('hh_standartize_mri_data: asset standartized z-score collection builded successfully')
    ### Collection of diagonales of group's z matrices for all groups:
    df_group_mean_z_diag = pd.concat(arr_group_diag_container, axis = 0, keys = arr_group_codes, names = ['Group Name', 'Date'], copy = False)
    print('hh_standartize_mri_data: data vector collection of diagonales of mean z score matrix for all groups builded successfully')       
    ### Collection of percentiled group's z matrices for all groups:
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

    ### Initialising containers for weighted mean matrix calculation:
    arr_matrix_container = []
    arr_group_codes = []
    ### Group aggregating cycle:    
    for group_index, ser_group_info in df_model_MRI.iterrows():
        group_code = ser_group_info['Asset Code']
        group_weight = ser_group_info['Factor Weights']
        ### Loading group z score matrix from HDF5 file:
        df_group_z_matrix = pd.read_hdf(hdf_z_matrix_path, group_code)
        df_group_z_matrix.set_index(0, drop = True, inplace = True)
        ### Initialising not np.NaN weights sum DataFrame:
        if (group_index == 0):
            df_group_weights = pd.DataFrame(np.zeros(df_group_z_matrix.shape), index = df_group_z_matrix.index, columns = df_group_z_matrix.columns)
        ### Adding weighted matrix to container:
        arr_matrix_container.append(df_group_z_matrix * group_weight)
        arr_group_codes.append(group_code)
        # Adding not np.NaN weights to aggregated DataFrame:
        df_group_weights = df_group_weights + df_group_z_matrix.notna() * group_weight
        print('hh_aggregate_mri_data: group', group_code, 'z matrix data extracted successfully')
    ### Calculating z matrix for MRI from weighted group matrices:        
    df_MRI_mean = pd.concat(arr_matrix_container, axis = 0, keys = arr_group_codes, names = ['Group Code', 'Date'], copy = False) 
    df_MRI_mean = df_MRI_mean.sum(level = 1)
    df_group_weights[df_group_weights == 0] = np.NaN
    df_MRI_mean = df_MRI_mean / df_group_weights
    print('hh_aggregate_mri_data: weighted mean matrix for MRI builded successfully')    
    df_MRI_mean_z = (df_MRI_mean - df_MRI_mean.mean())/df_MRI_mean.std()
    ser_MRI_mean_z_diag = pd.Series(np.copy(np.diag(df_MRI_mean_z)), index = df_MRI_mean_z.index)
    print('hh_aggregate_mri_data: z-score matrix for MRI weighted mean matrix builded successfully')     
    ### Initialising containers for weighted percentiled vector calculation:
    arr_perc_container = []    
    ### Group aggregating cycle:    
    for group_index_perc, ser_group_info_perc in df_model_MRI.iterrows():
        group_code = ser_group_info_perc['Asset Code']
        group_weight = ser_group_info_perc['Factor Weights']
        ### Loading group z scored matrix percentalization result from HDF5 file:
        ser_group_perc= pd.read_hdf(hdf_group_info_path, object_perc_grouped_hdf)[group_code]
        # Initialising not np.NaN weights sum TimeSeries:
        if (group_index_perc == 0):
            ser_group_weights = pd.Series(np.zeros(len(ser_group_perc)), index = ser_group_perc.index)        
        ### Adding weighted matrix to container:
        arr_perc_container.append(ser_group_perc * group_weight)        
        ### Adding not np.NaN weights to aggregated TimeSeries:
        ser_group_weights = ser_group_weights + ser_group_perc.notna() * group_weight
        print('hh_aggregate_mri_data: group', group_code, 'percentiled data vector extracted successfully')        
    ### Calculating z matrix for MRI from weighted group matrices:        
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
    
    ### Extracting asset level data:
    df_raw_data = pd.read_hdf(asset_hdf_path, asset_selected_key)
    df_raw_data.set_index('Date', drop = True, inplace = True)
    df_standart_data = pd.read_hdf(asset_hdf_path, asset_z_score_key)
    df_standart_data.set_index('Date', drop = True, inplace = True)
    ### Extracting group level data:
    df_group_diag = pd.read_hdf(group_hdf_path, group_diag_key)
    df_group_diag.set_index('Date', drop = True, inplace = True)
    df_group_perc = pd.read_hdf(group_hdf_path, group_perc_key)
    df_group_perc.set_index('Date', drop = True, inplace = True)
    ### Extracting MRI level data:
    ser_mri_diag = pd.read_hdf(MRI_hdf_path, mri_diag_key)
    ser_mri_raw_perc = pd.read_hdf(MRI_hdf_path, mri_raw_key)
    ser_mri_ma_perc = pd.read_hdf(MRI_hdf_path, mri_ma_key)
    ### Drawing subplots for each group:  
    dict_fig_MRI = {}
    for group_counter, (asset_group_name, df_asset_group) in enumerate(df_model_asset.groupby('Asset Group')):
        ### Initialising group visibility variables:
        fig_group, axes_group = plt.subplots(3, 1, figsize = figure_size)
        plt.subplots_adjust(hspace = figure_hspace)
        axes_group[0].set_title(asset_group_name + ' / raw series')
        axes_group[1].set_title(asset_group_name + ' / standartized winsorized z-scores of series')  
        axes_group[2].set_title(asset_group_name + ' / group weighted mean')        
        for (asset_index, asset_code) in df_asset_group['Asset Code'].iteritems():    
            ### Drawing raw series for current group assets:            
            df_raw_data[asset_code].plot(ax = axes_group[0], label = asset_code, alpha = 0.75, grid = True, rot = 0, linestyle = ':')
            ### Drawing z-score series for current group assets:                       
            df_standart_data[asset_code].plot(ax = axes_group[1], label = asset_code, alpha = 0.75, grid = True, rot = 0, linestyle = ':')
        ### Drawing z-score for group weighted mean:        
        df_group_diag[asset_group_name].plot(ax = axes_group[2], label = asset_group_name, alpha = 1.0, grid = True, rot = 0) 
        ### Configuring plots:
        for ax_group in axes_group:
            ax_group.legend(loc = 'best')
            ax_group.xaxis.set_major_locator(mdates.YearLocator())
            ax_group.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax_group.xaxis.set_minor_locator(mdates.MonthLocator()) 
            ax_group.set_xlim(min_date, max_date)             
        ### Adding plot to dictionary array:        
        dict_fig_MRI[asset_group_name] = fig_group    
    ### Drawing subplots for MRI:
    fig_MRI, axes_MRI = plt.subplots(3, 1, figsize = figure_size)    
    plt.subplots_adjust(hspace = figure_hspace)    
    axes_MRI[0].set_title('Separate groups and common MRI z-scores')    
    axes_MRI[1].set_title('Separate groups and common MRI percentiled values')  
    axes_MRI[2].set_title('MRI percentiled values: raw and by moving average')     
    for group_index, ser_group_info in df_model_MRI.iterrows():
        group_code = ser_group_info['Asset Code']
        ### Drawing z-score series for current group assets:
        df_group_diag[group_code].plot(ax = axes_MRI[0], label = group_code, alpha = 0.75, grid = True, rot = 0, linestyle = ':')        
        ### Drawing percentiled series for current group assets:        
        df_group_perc[group_code].plot(ax = axes_MRI[1], label = group_code, alpha = 0.75, grid = True, rot = 0, linestyle = ':')
    ### Drawing z-score series for MRI:
    ser_mri_diag.plot(ax = axes_MRI[0], label = 'MRI', alpha = 1.0, grid = True, rot = 0)    
    ### Drawing percentiled series for MRI:
    ser_mri_raw_perc.plot(ax = axes_MRI[1], label = 'MRI', alpha = 1.0, grid = True, rot = 0)    
    ### Drawing percentiled series for MRI:
    ser_mri_raw_perc.plot(ax = axes_MRI[2], label = 'Raw', alpha = 1.0, grid = True)
    ser_mri_ma_perc.plot(ax = axes_MRI[2], label = 'Moving average', alpha = 1.0, grid = True, rot = 0)
    ### Configuring plots:    
    axes_MRI[0].legend(loc = 'best')    
    axes_MRI[1].legend(loc = 'best')        
    axes_MRI[2].legend(loc = 'best')  
    for ax_MRI in axes_MRI:
        ax_MRI.legend(loc = 'best')
        ax_MRI.xaxis.set_major_locator(mdates.YearLocator())
        ax_MRI.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax_MRI.xaxis.set_minor_locator(mdates.MonthLocator()) 
        ax_MRI.set_xlim(min_date, max_date)
    ### Adding plot to dictionary array:    
    dict_fig_MRI['MRI'] = fig_MRI
        
    return dict_fig_MRI


def hh_bokeh_mri_data(df_model_asset, df_model_MRI, asset_hdf_path, group_hdf_path, MRI_hdf_path, asset_selected_key, asset_z_score_key, group_diag_key, group_perc_key, mri_diag_key, mri_raw_key, mri_ma_key, figure_size, bokeh_toolbar, min_date, max_date, df_risk_band):
    """
    Version 0.03 2019-04-16
    
    FUNCTIONALITY:
      1) Drawing plot for all groups
      2) Drawing plot for MRI
    OUTPUT:
      arr_figures (array) - array of group bokeh.plotting.figure and MRI bokeh.plotting.figure
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
    from bokeh.layouts import widgetbox
    from bokeh.layouts import column as b_col
    import bokeh.models as b_md    
    from bokeh.palettes import Set2, Set3
    import bokeh.plotting as b_pl

    ### Defining output to notebook or to html file:
    b_pl.output_notebook()
        
    ### Extracting asset level data:
    df_raw_data = pd.read_hdf(asset_hdf_path, asset_selected_key)
    df_raw_data.set_index('Date', drop = True, inplace = True)
    df_standart_data = pd.read_hdf(asset_hdf_path, asset_z_score_key)
    df_standart_data.set_index('Date', drop = True, inplace = True)
    ### Extracting group level data:
    df_group_diag = pd.read_hdf(group_hdf_path, group_diag_key)
    df_group_diag.set_index('Date', drop = True, inplace = True)
    df_group_perc = pd.read_hdf(group_hdf_path, group_perc_key)
    df_group_perc.set_index('Date', drop = True, inplace = True)
    ### Extracting MRI level data:
    ser_mri_diag = pd.read_hdf(MRI_hdf_path, mri_diag_key)
    ser_mri_raw_perc = pd.read_hdf(MRI_hdf_path, mri_raw_key)
    ser_mri_ma_perc = pd.read_hdf(MRI_hdf_path, mri_ma_key)

    ### Initialising of resulting array:
    arr_figures = []
    ### Initialising of x-axis array for all plots:
    arr_index_values = df_raw_data.index.values

    ### Drawing plots for asset groups:
    for group_counter, (asset_group_name, df_asset_group) in enumerate(df_model_asset.groupby('Asset Group')):
        ### Defining color palette for each group:
        num_asset_lines = len(df_asset_group['Asset Code'])
        palette_asset_lines = Set2[8][ : num_asset_lines]
        ### Initialising raw asset plot for group:
        fig_group_raw = b_pl.figure(tools = bokeh_toolbar, x_axis_label = 'Date', plot_width = figure_size[0], plot_height = figure_size[1], x_axis_type = 'datetime', 
                                    title = asset_group_name + ' / raw series', x_range = (min_date, max_date))
        ### Initialising z-score asset plot for group:        
        fig_group_z_score = b_pl.figure(tools = bokeh_toolbar, x_axis_label = 'Date', plot_width = figure_size[0], plot_height = figure_size[1], x_axis_type = 'datetime',
                                        title = asset_group_name + ' / standartized winsorized z-scores of series', x_range = fig_group_raw.x_range)        
        ### Drawing asset lines for asset plots:
        for asset_counter, asset_code in enumerate(df_asset_group['Asset Code']):
            fig_group_raw.line(x = arr_index_values, y = df_raw_data[asset_code].values,
                               line_color = palette_asset_lines[asset_counter], line_width = 1, legend = asset_code, name = str(asset_counter))   
            fig_group_z_score.line(x = arr_index_values, y = df_standart_data[asset_code].values,
                               line_color = palette_asset_lines[asset_counter], line_width = 1, legend = asset_code, name = str(asset_counter)) 
        ### Initialising and drawing consolidated line for group z-score:
        fig_group_mean = b_pl.figure(tools = bokeh_toolbar, x_axis_label = 'Date', plot_width = figure_size[0], plot_height = figure_size[1], x_axis_type = 'datetime',
                                     title = asset_group_name + ' / group weighted mean', x_range = fig_group_raw.x_range)
        fig_group_mean.line(x = arr_index_values, y = df_group_diag[asset_group_name].values, legend = asset_group_name)
        ### Tuning plots and adding risk event bands:
        fig_group_raw.legend.location = 'top_left'
        fig_group_raw.legend.background_fill_alpha  = 0.75
        fig_group_raw.legend.spacing  = 0        
        fig_group_raw.xaxis.axis_label_standoff = 0
        fig_group_raw.title.align = 'center'   
        fig_group_raw.min_border_bottom = 50
        for event_counter, ser_event in df_risk_band.iterrows():
            fig_group_raw.add_layout(b_md.BoxAnnotation(left = ser_event['Begin date'], right = ser_event['End date'], fill_alpha = 0.1, fill_color = 'red'))
        fig_group_raw.add_tools(b_md.HoverTool(tooltips = [('Date', '@x{%F}')], formatters = {'x': 'datetime'}, mode = 'vline', names = ['0']))            
        fig_group_z_score.legend.location = 'top_left'                
        fig_group_z_score.legend.background_fill_alpha  = 0.75        
        fig_group_z_score.legend.spacing  = 0   
        fig_group_z_score.xaxis.axis_label_standoff = 0        
        fig_group_z_score.title.align = 'center'    
        fig_group_z_score.min_border_bottom = 50
        for event_counter, ser_event in df_risk_band.iterrows():
            fig_group_z_score.add_layout(b_md.BoxAnnotation(left = ser_event['Begin date'], right = ser_event['End date'], fill_alpha = 0.1, fill_color = 'red'))        
        fig_group_z_score.add_tools(b_md.HoverTool(tooltips = [('Date', '@x{%F}')], formatters = {'x': 'datetime'}, mode = 'vline', names = ['0']))             
        fig_group_mean.legend.location = 'top_left'
        fig_group_mean.legend.background_fill_alpha  = 0.75  
        fig_group_mean.legend.spacing  = 0        
        fig_group_mean.xaxis.axis_label_standoff = 0
        fig_group_mean.title.align = 'center'
        fig_group_mean.min_border_bottom = 50        
        for event_counter, ser_event in df_risk_band.iterrows():
            fig_group_mean.add_layout(b_md.BoxAnnotation(left = ser_event['Begin date'], right = ser_event['End date'], fill_alpha = 0.1, fill_color = 'red'))
        fig_group_mean.add_tools(b_md.HoverTool(tooltips = [('Date', '@x{%F}')], formatters = {'x': 'datetime'}, mode = 'vline'))
        ### Conslidating plots for asset group and adding layout to resulting plot:
        arr_figures.append(b_col(fig_group_raw, fig_group_z_score, fig_group_mean))
    ### Defining color palette for MRI:
    num_group_lines = len(df_model_MRI['Asset Code'])
    palette_group_lines = Set2[8][ : num_group_lines]  
    ### Initialising z-score and percentiled values plots for MRI:
    fig_MRI_z_score = b_pl.figure(tools = bokeh_toolbar, x_axis_label = 'Date', plot_width = figure_size[0], plot_height = figure_size[1], x_axis_type = 'datetime',
                                  title = 'Separate groups and common MRI z-scores', x_range = (min_date, max_date))  
    fig_MRI_perc = b_pl.figure(tools = bokeh_toolbar, x_axis_label = 'Date', plot_width = figure_size[0], plot_height = figure_size[1], x_axis_type = 'datetime',
                               title = 'Separate groups and common MRI percentiled values', x_range = fig_MRI_z_score.x_range)  
    ### Drawing lines for groups:
    for group_counter, group_code in enumerate(df_model_MRI['Asset Code']):
            fig_MRI_z_score.line(x = arr_index_values, y = df_group_diag[group_code].values, 
                                 line_color = palette_group_lines[group_counter], line_width = 1, legend = group_code) 
            fig_MRI_perc.line(x = arr_index_values, y = df_group_perc[group_code].values,
                              line_color = palette_group_lines[group_counter], line_width = 1, legend = group_code, name = str(group_counter))             
    ### Drawing line for MRI:
    fig_MRI_z_score.line(x = arr_index_values, y = ser_mri_diag.values, legend = 'MRI z score', line_width = 2, name = '0')
    ### Drawing line for MRI percentiled:
    fig_MRI_perc.line(x = arr_index_values, y = ser_mri_raw_perc.values, legend = 'MRI percentile', color = 'blue')    
    ### Initialising plot for MRI percentiled:
    fig_MRI_res = b_pl.figure(tools = bokeh_toolbar, x_axis_label = 'Date', plot_width = figure_size[0], plot_height = figure_size[1], x_axis_type = 'datetime',
                              title = 'MRI percentiled values: raw and by moving average', x_range = fig_MRI_z_score.x_range)      
    ### Drawing lines for MRI percentiled (raw and moving average):
    fig_MRI_res.line(x = arr_index_values, y = ser_mri_raw_perc.values, legend = 'Raw', color = 'blue', name = '0')   
    fig_MRI_res.line(x = arr_index_values, y = ser_mri_ma_perc.values, legend = 'MA-5', color = 'orange') 
    ### Tuning plots and adding risk event bands:
    fig_MRI_z_score.legend.location = 'top_left'
    fig_MRI_z_score.legend.background_fill_alpha  = 0.75    
    fig_MRI_z_score.legend.spacing  = 0        
    fig_MRI_z_score.xaxis.axis_label_standoff = 0
    fig_MRI_z_score.title.align = 'center'  
    fig_MRI_z_score.min_border_bottom = 50 
    for event_counter, ser_event in df_risk_band.iterrows():
        fig_MRI_z_score.add_layout(b_md.BoxAnnotation(left = ser_event['Begin date'], right = ser_event['End date'], fill_alpha = 0.1, fill_color = 'red'))    
    fig_MRI_z_score.add_tools(b_md.HoverTool(tooltips = [('Date', '@x{%F}')], formatters = {'x': 'datetime'}, mode = 'vline', names = ['0']))          
    fig_MRI_perc.legend.location = 'top_left'
    fig_MRI_perc.legend.background_fill_alpha  = 0.75
    fig_MRI_perc.legend.spacing  = 0        
    fig_MRI_perc.xaxis.axis_label_standoff = 0
    fig_MRI_perc.title.align = 'center'
    fig_MRI_perc.min_border_bottom = 50    
    for event_counter, ser_event in df_risk_band.iterrows():
        fig_MRI_perc.add_layout(b_md.BoxAnnotation(left = ser_event['Begin date'], right = ser_event['End date'], fill_alpha = 0.1, fill_color = 'red'))   
    fig_MRI_perc.add_tools(b_md.HoverTool(tooltips = [('Date', '@x{%F}')], formatters = {'x': 'datetime'}, mode = 'vline', names = ['0']))          
    fig_MRI_res.legend.location = 'top_left'
    fig_MRI_res.legend.background_fill_alpha  = 0.75
    fig_MRI_res.legend.spacing  = 0       
    fig_MRI_res.xaxis.axis_label_standoff = 0    
    fig_MRI_res.title.align = 'center'
    fig_MRI_res.min_border_bottom = 50     
    for event_counter, ser_event in df_risk_band.iterrows():
        fig_MRI_res.add_layout(b_md.BoxAnnotation(left = ser_event['Begin date'], right = ser_event['End date'], fill_alpha = 0.1, fill_color = 'red'))     
    fig_MRI_res.add_tools(b_md.HoverTool(tooltips = [('Date', '@x{%F}')], formatters = {'x': 'datetime'}, mode = 'vline', names = ['0']))    
    ### Consloidating plots for asset group and adding layout to resulting plotЖ
    arr_figures.append(b_col(fig_MRI_z_score, fig_MRI_perc, fig_MRI_res))       
        
    return arr_figures


def hh_bokeh_markets_universe_data(df_history, figure_size, figure_title):
    """
    Version 0.04 2019-04-30
    
    FUNCTIONALITY:
      Drawing plot for markets membership
    OUTPUT:
      bokeh.layouts.column that consists of:
          range_slider_period (bokeh.models.widgets.DateRangeSlider) - slider to date period choosing for plot investigating purposes
          fig_history (bokeh.plotting.figure) - figure to display
    INPUT:
      df_history (pd.DataFrame) - history data set for plotting
      figure_size (tuple) - figure shapes
      figure_title (string) - figure title
    """
    
    import numpy as np
    import pandas as pd
    from datetime import datetime 
    from bokeh.palettes import Set2, Set3
    import bokeh.plotting as b_pl
    import bokeh.models as b_md
    from bokeh.layouts import widgetbox    
    from bokeh.layouts import row as b_row
    from bokeh.layouts import column as b_col
    from bokeh.transform import factor_cmap
    from bokeh.models.widgets import DateRangeSlider
    from bokeh.models import CustomJS

    ### Defining output to notebook or to html file:
    b_pl.output_notebook()        
    date_format = '%Y-%m-%d'
    ### Defining borders for x axe:
    date_left_border = df_history['Start Date'].min()
    date_right_border = df_history['End Date'].max()
    ### Initialising figure:
    fig_history = b_pl.figure(tools = ['box_zoom, reset'], plot_width = figure_size[0], plot_height = figure_size[1], title = figure_title,
                              x_axis_label = 'Date', x_axis_type = 'datetime', x_range = (date_left_border, date_right_border),
                              y_range = df_history.sort_index(ascending = False).index.unique().to_numpy())
    ### Range Slider tuning:
    slider_callback = CustomJS(args = dict(fig_to_update = fig_history), 
                               code = """
                                      var slider_period = cb_obj.value;
                                      fig_to_update.x_range.start = slider_period[0];
                                      fig_to_update.x_range.end = slider_period[1];
                                      fig_to_update.change.emit();
                                      """)
    range_slider_period = DateRangeSlider(start = date_left_border, end = date_right_border, value = (date_left_border, date_right_border), 
                                          title = "Date period to display", width = (figure_size[1] - 25), bar_color = 'gray', step = 1)
    range_slider_period.js_on_change('value', slider_callback)
    ### Iterating market classes to allow muting:
    for index_counter, history_index in enumerate(df_history['Index Name'].unique()):
        ### Defining DataSource for plotting:
        cds_membership = b_md.ColumnDataSource(df_history[df_history['Index Name'] == history_index])    
        ### Drawing horizontal bars for all DataFrame rows at once:
        fig_history.hbar(source = cds_membership, y = 'Member Code', left = 'Start Date', right = 'End Date',
                         height = 1, line_color = 'white', fill_color = Set2[4][index_counter], legend = 'Index Name',
                         muted_alpha = 0.25, muted_line_color = 'white', muted_color = Set2[4][index_counter])
    ### Configuring output:
    fig_history.legend.location = 'top_left'
    fig_history.legend.click_policy = 'mute'
    fig_history.add_tools(b_md.HoverTool(tooltips = [('Country', '@Country'), ('([ , ])', '(@{Start Date}{%F}, @{End Date}{%F})'), ('MSCI Index', '@{Index Name}')],
                                         formatters = {'Start Date': 'datetime', 'End Date': 'datetime'}))
    fig_history.toolbar.autohide = True         
    return b_col(widgetbox(range_slider_period), fig_history)


def hh_bokeh_compare_universe_data(df_compare, figure_size, figure_title):
    """
    Version 0.01 2019-04-30
    
    FUNCTIONALITY:
      Drawing plot for comparing markets membership in MSCI and ISON universes
    OUTPUT:
      bokeh.layouts.column that consists of:
          range_slider_period (bokeh.models.widgets.DateRangeSlider) - slider to date period choosing for plot investigating purposes
          fig_compare (bokeh.plotting.figure) - figure to display
    INPUT:
      df_compare (pd.DataFrame) - result of comparision of data sets for plotting
      figure_size (tuple) - figure shapes
      figure_title (string) - figure title
    """
    
    import numpy as np
    import pandas as pd
    from datetime import datetime 
    from bokeh.palettes import Set2, Set3
    import bokeh.plotting as b_pl
    import bokeh.models as b_md
    from bokeh.layouts import widgetbox    
    from bokeh.layouts import row as b_row
    from bokeh.layouts import column as b_col
    from bokeh.transform import factor_cmap
    from bokeh.models.widgets import DateRangeSlider
    from bokeh.models import CustomJS

    ### Defining output to notebook or to html file:
    b_pl.output_notebook()        
    date_format = '%Y-%m-%d'
    ### Defining borders for x axe:
    date_left_border = df_compare['Start Date'].min()
    date_right_border = df_compare['End Date'].max()
    ### Initialising figure:
    fig_compare = b_pl.figure(tools = ['box_zoom, reset'], plot_width = figure_size[0], plot_height = figure_size[1], title = figure_title,
                              x_axis_label = 'Date', x_axis_type = 'datetime', x_range = (date_left_border, date_right_border),
                              y_range = df_compare['Member Code'].sort_values(ascending = False).unique())
    ### Range Slider tuning:
    slider_callback = CustomJS(args = dict(fig_to_update = fig_compare), 
                               code = """
                                      var slider_period = cb_obj.value;
                                      fig_to_update.x_range.start = slider_period[0];
                                      fig_to_update.x_range.end = slider_period[1];
                                      fig_to_update.change.emit();
                                      """)
    range_slider_period = DateRangeSlider(start = date_left_border, end = date_right_border, value = (date_left_border, date_right_border), 
                                          title = "Date period to display", width = 975, bar_color = 'gray', step = 1)
    range_slider_period.js_on_change('value', slider_callback)

    ### Iterating market classes to allow muting:
    for status_counter, compare_status in enumerate(df_compare['Status'].unique()):
    ### Defining DataSource for plotting:
        cds_comparision = b_md.ColumnDataSource(df_compare[df_compare['Status'] == compare_status])
        ### Drawing horizontal bars for all DataFrame rows at once:
        fig_compare.hbar(source = cds_comparision, y = 'Member Code', left = 'Start Date', right = 'End Date',
                         height = 1, line_color = 'white', fill_color = Set2[4][status_counter], legend = 'Status',
                         muted_alpha = 0.25, muted_line_color = 'white', muted_color = Set2[4][status_counter])
    ### Configuring output:
    fig_compare.legend.location = 'top_left'
    fig_compare.legend.click_policy = 'mute'
    fig_compare.add_tools(b_md.HoverTool(tooltips = [('Country', '@Country'), ('([ , ])', '(@{Start Date}{%F}, @{End Date}{%F})'), ('MSCI Index', '@{Index Name}')],
                                         formatters = {'Start Date': 'datetime', 'End Date': 'datetime'}))
    fig_compare.toolbar.autohide = True  
    
    return b_col(widgetbox(range_slider_period), fig_compare)

def hh_bokeh_MSCI_country_returns(df_current, df_history, df_USD_pivot, df_LOC_pivot, figure_size, figure_title):
    """
    Version 0.01 2019-05-02
    
    FUNCTIONALITY:
      Drawing plot for illustrating MSCI returns for choosed country
    OUTPUT:
      bokeh.layouts.column that consists of:
          radio_order (bokeh.models.widgets.RadioGroup) - radio menu to choose country sorting order
          select_country (bokeh.models.widgets.Select) - list of countries to select          
          fig_returns (bokeh.plotting.figure) - figure to display
          range_slider_period (bokeh.models.widgets.DateRangeSlider) - slider to date period choosing for plot investigating purposes
    INPUT:
      df_current (pd.DataFrame) - current membership of MSCI
      df_history (pd.DataFrame) - MSCI membership history
      df_USD_pivot (pd.DataFrame) - country returns data in USD
      df_LOC_pivot (pd.DataFrame) - country returns data in local currency      
      figure_size (tuple) - figure shapes
      figure_title (string) - figure title
    """

    import numpy as np
    import pandas as pd
    from datetime import datetime 
    from bokeh.palettes import Set2, Set3
    import bokeh.plotting as b_pl
    import bokeh.models as b_md
    from bokeh.layouts import widgetbox    
    from bokeh.layouts import row as b_row
    from bokeh.layouts import column as b_col
    from bokeh.layouts import widgetbox
    from bokeh.transform import factor_cmap
    from bokeh.models.widgets import Dropdown, Select, RadioGroup, DateRangeSlider
    from bokeh.models import CustomJS

    ### Defining output to notebook or to html file:
    b_pl.output_notebook()
    ### Defining initial country:
    code_initial = 'AR'
    ### Preparing date output format:
    date_format = '%Y-%m-%d'
    ### Creating sorting orders names:
    arr_labels = []
    arr_labels.append('Sort by: Country Name')
    arr_labels.append('Sort by: Country Code')
    arr_labels.append('Sort by: Market + Country Name')
    arr_labels.append('Sort by: Market + Country Code')
    ### Creating sorted by different sorting orders country lists:
    arr_sortings = []
    arr_sortings.append(list(zip(list(df_current.sort_values(by = 'Country')['Member Code']), 
                                 list(df_current.sort_values(by = 'Country')['Search Name']))))
    arr_sortings.append(list(zip(list(df_current.sort_values(by = 'Member Code')['Member Code']), 
                                 list(df_current.sort_values(by = 'Member Code')['Search Name']))))
    arr_sortings.append(list(zip(list(df_current.sort_values(by = ['Index Name', 'Country'])['Member Code']), 
                                 list(df_current.sort_values(by = ['Index Name', 'Country'])['Search Name']))))
    arr_sortings.append(list(zip(list(df_current.sort_values(by = ['Index Name', 'Member Code'])['Member Code']), 
                                 list(df_current.sort_values(by = ['Index Name', 'Member Code'])['Search Name']))))
    ### Initial country defining:
    df_USD_pivot['YY'] = df_USD_pivot[code_initial]
    df_LOC_pivot['YY'] = df_LOC_pivot[code_initial]
    ### DataSource defining:
    cds_USD_returns = b_md.ColumnDataSource(df_USD_pivot)
    cds_LOC_returns = b_md.ColumnDataSource(df_LOC_pivot)
    ### Initial x axis boundaries defining:
    date_left_border = df_USD_pivot['YY'].idxmin()
    date_right_border = df_USD_pivot['YY'].idxmax()
    ### Creating main figure:
    fig_returns = b_pl.figure(tools = ['pan, box_zoom, reset'], plot_width = figure_size[0], plot_height = figure_size[1], 
                              title = figure_title + ' (' + code_initial + ')',
                              x_axis_label = 'Date', x_axis_type = 'datetime', x_range = (date_left_border, date_right_border))
    ### Drawing plots:
    line_USD = fig_returns.line(source = cds_USD_returns, x = 'Date', y = 'YY', legend = 'USD', color = 'green', line_width = 2, name = 'USD')
    line_LOC = fig_returns.line(source = cds_LOC_returns, x = 'Date', y = 'YY', legend = 'LOCAL', color = 'blue', line_width = 2, name = 'LOC')
    ### Configuring output:
    fig_returns.legend.location = 'top_left' 
    fig_returns.add_tools(b_md.HoverTool(tooltips = [('Value', '@YY{0,0.00}'), ('Date', '@Date{%F}')], formatters = {'Date': 'datetime', 'YY': 'numeral'})) 
    fig_returns.toolbar.autohide = True    
    ### Constant maximum quantity of reclassificationsa for one country:
    num_annotations = 5
    ### Creating array of blank annotations:
    arr_annotations = []
    for counter_annotation in np.arange(num_annotations):
        arr_annotations.append(b_md.BoxAnnotation(left = date_right_border, right = date_right_border, fill_alpha = 0.2, fill_color = 'white'))
        fig_returns.add_layout(arr_annotations[counter_annotation])
    ### Creating disctionary for annotations colors:
    dict_ann_colors = {'DM': Set2[4][0], 'EM': Set2[4][1], 'FM': Set2[4][2], 'SM': Set2[4][3]}
    ### Creating datasource for annotations:
    cds_annotations = b_md.ColumnDataSource(df_history)
    ### Drawing annotations for initial country:
    counter_annotations = 0
    for reclass_index, ser_reclass in df_history.iterrows():
        if (reclass_index == code_initial):
            arr_annotations[counter_annotations].fill_color = dict_ann_colors[ser_reclass['Index Name']];
            arr_annotations[counter_annotations].left = ser_reclass['Start Date'];
            arr_annotations[counter_annotations].right = ser_reclass['End Date'];
            counter_annotations = counter_annotations + 1;
    ### Range Slider for date period tuning (including connection betwwen slider values and figure boundaries):
    slider_callback = CustomJS(args = dict(fig_to_update = fig_returns), 
                               code = """
                                      var slider_period = cb_obj.value;
                                      fig_to_update.x_range.start = slider_period[0];
                                      fig_to_update.x_range.end = slider_period[1];
                                      fig_to_update.change.emit();
                                      """)
    range_slider_period = DateRangeSlider(start = date_left_border, end = date_right_border, value = (date_left_border, date_right_border), 
                                          title = "Date period to display", width = figure_size[0] - 25, bar_color = 'gray', step = 1)
    range_slider_period.js_on_change('value', slider_callback)
    ### Select for choosing country tuning (including connection between selected value and range slider boundaries, figure plots, figure boundaries, figure title):
    select_callback = CustomJS(args = dict(fig_to_update = fig_returns, figure_title = figure_title, widget_to_update = range_slider_period,
                                           cds_USD = cds_USD_returns, cds_LOC = cds_LOC_returns, cds_annotations = cds_annotations,
                                           date_max = date_right_border, 
                                           arr_annotations = arr_annotations, dict_colors = dict_ann_colors, num_annotations = num_annotations),
                               code = """
                                      var chosen_country = cb_obj.value;    
                                      cds_USD.data['YY'] = cds_USD.data[chosen_country];
                                      cds_LOC.data['YY'] = cds_LOC.data[chosen_country];                                  
                                      var date_min = cds_USD.data['Date'][0]                                 
                                      for (var i = 1; i <= cds_USD.get_length(); i++)
                                      {
                                        if (!isNaN(cds_USD.data['YY'][i]))
                                        {
                                          date_min = cds_USD.data['Date'][i];
                                          break;
                                        }
                                      }
                                      widget_to_update.start = date_min;
                                      widget_to_update.end = date_max;
                                      widget_to_update.value[0] = date_min;
                                      widget_to_update.value[1] = date_max;                               
                                      widget_to_update.change.emit();
                                      cds_USD.change.emit();
                                      cds_LOC.change.emit();
                                      var counter_ann = 0;
                                      for (var counter_ann = 0; counter_ann < num_annotations; counter_ann++)
                                      {
                                        arr_annotations[counter_ann].fill_color = 'white';
                                        arr_annotations[counter_ann].left = date_max;
                                        arr_annotations[counter_ann].right = date_max;
                                      }
                                      counter_ann = 0;                                  
                                      for (var i = 0; i <= cds_annotations.get_length(); i++)
                                      {
                                        if (cds_annotations.data['Member Code'][i] == chosen_country)
                                        {
                                          arr_annotations[counter_ann].fill_color = dict_colors[cds_annotations.data['Index Name'][i]];
                                          arr_annotations[counter_ann].left = cds_annotations.data['Start Date'][i];
                                          arr_annotations[counter_ann].right = cds_annotations.data['End Date'][i];
                                          counter_ann = counter_ann + 1;
                                        }
                                      }                             
                                      fig_to_update.x_range.start = date_min;
                                      fig_to_update.x_range.end = date_max;                                  
                                      fig_to_update.title.text = figure_title + ' (' + chosen_country + ')';
                                      fig_to_update.change.emit();
                                      """)
    select_country = Select(title = 'Country to show returns:', options = arr_sortings[0], callback = select_callback)
    ### Radio Group for sorting order tuning (including connection betwwen active radio and slider country list):
    radio_callback = CustomJS(args = dict(widget_to_update = select_country, order_source = arr_sortings),
                              code = """
                                     var sort_order = cb_obj.active;
                                     widget_to_update.options = order_source[sort_order];                             
                                     """)
    radio_order = RadioGroup(labels = arr_labels, active = 0, callback = radio_callback)

    return b_col(widgetbox(radio_order), widgetbox(select_country), fig_returns, widgetbox(range_slider_period))


def hh_msci_mri_regression(window_length, half_life_period, weights_distribution, MRI_delta_way, MRI_hdf_path, MRI_key, MSCI_returns_path, MSCI_returns_key):
    """
    Version 0.04 2019-05-23
    
    FUNCTIONALITY: 
      1) Creating weights vector for WLS performing
      2) Choosing rollback data vectors for each country/date pair
      3) Checking returns data vectors for enough intersection with corresponding MRI rollback data vector
      4) Performing WLS for calculating lenear regression parameters
    OUTPUT:
      df_wls_results (pd.DataFrame) - results of WLS performing for each attested country/date pair
    INPUT:
      window_length (integer) - maximal length of rolling back window to select data vector (in years)
      half_life_period (integer) - period of weight downcome to 1/2 before normalising (in months)    
      weights_distribution (string) - way of tuning weights for WLS:
        'none' - OLS or equal weights for all residual squares;
        'by_date' - descending order from point date down to window_length earlier;
        'by_value' - descending order depending on similarity (absolute difference) to date point value for regressor (MRI);        
      MRI_delta_way (string) - way of taking monthly delta for regressor:
        'point' - delta of end-of-month values;
        'mean' - delta of monthly means;         
      MRI_hdf_path (string) - path to the MRI released HDF5 file
      MRI_key (string) - data object key to access MRI values from HDF5 file
      MSCI_returns_path (string) - path to the MSCI returns HDF5 file
      MSCI_returns_key (string) - data object key to accessMSCI USD monthly returns from HDF5 file
    """    
    import numpy as np
    import pandas as pd    
    import math   
    import statsmodels.api as sm    
    ### Expanding visibility zone for Python engine to make HH Modules seen:
    import sys 
    sys.path.append('../..')    
    
    ### Defining period weights for future regressions:
    num_year_work_days = 252
    num_year_months = 12
    ### Array of half-life periods (in months):
    arr_half_life_periods = [1, 2, 3, 6, 12, 24]
    ### Regression window (in years):
    num_regression_length = window_length
    ### Array of regressioon window day numbers descending:
    arr_weight_days = np.arange(num_year_work_days * num_regression_length, 0, -1) - 1
    ### Dictionary of weight arrays for each half-life period:
    dict_weights_daily = {}
    for month_period in arr_half_life_periods:
        num_period_factor = math.exp(math.log(0.5) / round((num_year_work_days / num_year_months * month_period)))
        dict_weights_daily[month_period] = np.exp(math.log(num_period_factor) * arr_weight_days)
    ### Weights adopting for monthly returns format:
    ts_weights = pd.Series(dict_weights_daily[half_life_period][:: -21][::-1])
    ts_weights = ts_weights / ts_weights.sum()       
    ### Preparing data for regressions calculation:
    ser_MRI_main = pd.read_hdf(MRI_hdf_path, MRI_key)
    ### MRI adopting for monthly returns format:
    ser_MRI_monthly_mean = ser_MRI_main.groupby(pd.Grouper(freq = 'BM')).mean()    
    ser_MRI_monthly_point = ser_MRI_main.groupby(pd.Grouper(freq = 'BM')).last()    
    if (MRI_delta_way == 'mean'):
        ser_MRI_monthly_delta = (ser_MRI_monthly_mean - ser_MRI_monthly_mean.shift(1))
    if (MRI_delta_way == 'point'):
        ser_MRI_monthly_delta = (ser_MRI_monthly_point - ser_MRI_monthly_point.shift(1))        
    ### Preparing MSCI data for looping:
    df_returns = pd.read_hdf(MSCI_returns_path, MSCI_returns_key)    
    df_returns.reset_index(level = 'Country', drop = True, inplace = True)
    df_returns.drop('INDEX', level = 'Code', inplace = True)
    df_returns = df_returns.swaplevel()
    df_returns.sort_index(level = [0, 1], inplace = True)
    ser_returns = df_returns.squeeze()   
    ### From based 100 returns to month-to-month returns
    for country_iter in ser_returns.index.get_level_values(level = 0).unique():
        ser_returns[country_iter] = (ser_returns[country_iter] / ser_returns[country_iter].shift(1) - 1)
    ser_returns.fillna(0, inplace = True)
    ### Flattening MSCI changes by logarythm
    ser_returns = np.log(1 + ser_returns)       
    ### Regression loop performing:
    arr_index_container = []
    arr_results_container = []
    counter_points = 0
    counter_regressions = 0
    for (str_country_iter, date_iter) in ser_returns.index:
        counter_points = counter_points + 1
        ### Extracting returns data vector for each country/date point:
        ser_returns_iter = ser_returns[str_country_iter][(date_iter - pd.offsets.BMonthEnd(num_regression_length * 12 - 1)) : date_iter].dropna()
        ### Centralising returns data vector:
        ser_returns_iter = ser_returns_iter - ser_returns_iter.mean()
        ### Extacting MRI data vector for current date point:
        ser_MRI_values_iter = ser_MRI_monthly_mean[(date_iter - pd.offsets.BMonthEnd(num_regression_length * 12 - 1)) : date_iter].dropna()
        ser_MRI_iter = ser_MRI_monthly_delta[(date_iter - pd.offsets.BMonthEnd(num_regression_length * 12 - 1)) : date_iter].dropna()
        ### Centralising MRI data vector        
        ser_MRI_iter = ser_MRI_iter - ser_MRI_iter.mean()
        ### Intersecting MSCI and MRI indexes to select common part:
        index_iter = ser_MRI_iter.index.intersection(ser_returns_iter.index)
        ### Checking of enough data values to perform regression for current country/date point
        if (index_iter.size >= num_regression_length * 12 // 2):            
            counter_regressions = counter_regressions + 1
            ### Attaching weights vector to current date point:
            if (weights_distribution == 'none'):
                ser_weights = pd.Series((np.ones(num_regression_length * 12) / (num_regression_length * 12)), 
                                        index = pd.date_range(end = date_iter, periods = (num_regression_length * 12), freq = 'BM'))
            if (weights_distribution == 'by_date'):
                ser_weights = pd.Series(ts_weights.values, index = pd.date_range(end = date_iter, periods = (num_regression_length * 12), freq = 'BM'))
            if (weights_distribution == 'by_value'):       
                ser_weights = pd.Series(ts_weights.values, index = pd.date_range(end = date_iter, periods = (num_regression_length * 12), freq = 'BM'))
                index_weights = ser_MRI_values_iter.index.intersection(ser_weights.index)  
                ser_weights = ser_weights[index_weights]
                ser_distribution = ser_MRI_values_iter.copy()
                ser_distribution = abs(ser_distribution - ser_distribution.iloc[-1])
                ser_distribution.sort_values(ascending = False, inplace = True)
                df_distribution = ser_distribution.to_frame('From_Last')
                df_distribution['Weights'] = ser_weights.values
                df_distribution.drop(columns = ['From_Last'], axis = 1, inplace = True)
                ser_distribution = df_distribution.squeeze()
                ser_weights = ser_distribution.sort_index()          
            ### Regression performing:
            wls_y = ser_returns_iter[index_iter].values
            wls_x = ser_MRI_iter[index_iter].values
            wls_x = sm.add_constant(wls_x)
            wls_w = ser_weights[index_iter].values        
            wls_model = sm.WLS(wls_y, wls_x, weights = wls_w)            
            wls_results = wls_model.fit()
            arr_results_container.append(np.concatenate((wls_results.params, wls_results.tvalues, [wls_results.rsquared, wls_results.rsquared_adj])))      
            arr_index_container.append((str_country_iter, date_iter))
    ### Results index creating:
    index_wls_results = pd.MultiIndex.from_tuples(arr_index_container, names = ('Country', 'DatePoint'))
    ### Results aggregating:
    df_wls_results = pd.DataFrame(arr_results_container, index = index_wls_results, 
                                  columns = ['Beta_Const', 'Beta_MRI', 'T_Stat_Const', 'T_Stat_MRI', 'R_Squared', 'R_Squared_Adj'])
    
    print('hh_msci_mri_regression:', counter_points, 'MSCI monthly returns country/date points taken for regression on MRI.')     
    print('hh_msci_mri_regression:', counter_regressions, 'MSCI on MRI correct intersections choosed for regression performing')
    print('hh_msci_mri_regression: MSCI on MRI regression with weights distribution option "', weights_distribution, 
          '" and regressor monthly delta way "', MRI_delta_way,'" performed successfully.')
    
    return df_wls_results


def hh_msci_expvol(window_length, half_life_period, weights_distribution, MRI_hdf_path, MRI_key, MSCI_returns_path, MSCI_returns_key):
    """
    Version 0.02 2019-05-23
    
    FUNCTIONALITY: 
      1) Creating weights vector for WLS performing
      2) Choosing rollback data vectors for each country/date pair
      3) Checking returns data vectors for enough intersection with corresponding MRI rollback data vector
      4) Calculating exponential volatility for country/date pairs
    OUTPUT:
      df_wls_results (pd.DataFrame) - results of WLS performing for each attested country/date pair
    INPUT:
      window_length (integer) - maximal length of rolling back window to select data vector (in years)
      half_life_period (integer) - period of weight downcome to 1/2 before normalising (in months)    
      weights_distribution (string) - way of tuning weights for WLS:
        'none' - OLS or equal weights for all residual squares;
        'by_date' - descending order from point date down to window_length earlier;
        'by_value' - descending order depending on similarity (absolute difference) to date point value for regressor (MRI);               
      MRI_hdf_path (string) - path to the MRI released HDF5 file
      MRI_key (string) - data object key to access MRI values from HDF5 file
      MSCI_returns_path (string) - path to the MSCI returns HDF5 file
      MSCI_returns_key (string) - data object key to accessMSCI USD monthly returns from HDF5 file
    """    
    import numpy as np
    import pandas as pd    
    import math   
    import statsmodels.api as sm    
    ### Expanding visibility zone for Python engine to make HH Modules seen:
    import sys 
    sys.path.append('../..')    
    
    ### Defining period weights for future regressions:
    num_year_work_days = 252
    num_year_months = 12
    ### Array of half-life periods (in months):
    arr_half_life_periods = [1, 2, 3, 6, 12, 24]
    ### Regression window (in years):
    num_regression_length = window_length
    ### Array of regressioon window day numbers descending:
    arr_weight_days = np.arange(num_year_work_days * num_regression_length, 0, -1) - 1
    ### Dictionary of weight arrays for each half-life period:
    dict_weights_daily = {}
    for month_period in arr_half_life_periods:
        num_period_factor = math.exp(math.log(0.5) / round((num_year_work_days / num_year_months * month_period)))
        dict_weights_daily[month_period] = np.exp(math.log(num_period_factor) * arr_weight_days)
    ### Weights adopting for monthly returns format:
    ts_weights = pd.Series(dict_weights_daily[half_life_period][:: -21][::-1])
    ts_weights = ts_weights / ts_weights.sum()       
    ### Preparing data for regressions calculation:
    ser_MRI_main = pd.read_hdf(MRI_hdf_path, MRI_key)
    ### MRI adopting for monthly returns format:
    ser_MRI_monthly_mean = ser_MRI_main.groupby(pd.Grouper(freq = 'BM')).mean()          
    ### Preparing MSCI data for looping:
    df_returns = pd.read_hdf(MSCI_returns_path, MSCI_returns_key)    
    df_returns.reset_index(level = 'Country', drop = True, inplace = True)
    df_returns.drop('INDEX', level = 'Code', inplace = True)
    df_returns = df_returns.swaplevel()
    df_returns.sort_index(level = [0, 1], inplace = True)
    ser_returns = df_returns.squeeze()   
    ### From based 100 returns to month-to-month returns
    for country_iter in ser_returns.index.get_level_values(level = 0).unique():
        ser_returns[country_iter] = (ser_returns[country_iter] / ser_returns[country_iter].shift(1) - 1)
    ser_returns.fillna(0, inplace = True)
    ### Flattening MSCI changes by logarythm
    ser_returns = np.log(1 + ser_returns)    
    ### Regression loop performing:
    arr_index_container = []
    arr_results_container = []
    counter_points = 0
    counter_regressions = 0
    for (str_country_iter, date_iter) in ser_returns.index:
        counter_points = counter_points + 1
        ### Extracting returns data vector for each country/date point:
        ser_returns_iter = ser_returns[str_country_iter][(date_iter - pd.offsets.BMonthEnd(num_regression_length * 12 - 1)) : date_iter].dropna()
        ### Centralising returns data vector:
        ser_returns_iter = ser_returns_iter - ser_returns_iter.mean()
        ### Extacting MRI data vector for current date point:
        ser_MRI_values_iter = ser_MRI_monthly_mean[(date_iter - pd.offsets.BMonthEnd(num_regression_length * 12 - 1)) : date_iter].dropna()
        ### Intersecting MSCI and MRI indexes to select common part:
        index_iter = ser_MRI_values_iter.index.intersection(ser_returns_iter.index)
        ### Checking of enough data values to perform regression for current country/date point
        if (index_iter.size >= num_regression_length * 12 // 2):            
            counter_regressions = counter_regressions + 1
            ### Attaching weights vector to current date point:
            if (weights_distribution == 'none'):
                ser_weights = pd.Series((np.ones(num_regression_length * 12) / (num_regression_length * 12)), 
                                        index = pd.date_range(end = date_iter, periods = (num_regression_length * 12), freq = 'BM'))
            if (weights_distribution == 'by_date'):
                ser_weights = pd.Series(ts_weights.values, index = pd.date_range(end = date_iter, periods = (num_regression_length * 12), freq = 'BM'))
            if (weights_distribution == 'by_value'):       
                ser_weights = pd.Series(ts_weights.values, index = pd.date_range(end = date_iter, periods = (num_regression_length * 12), freq = 'BM'))
                index_weights = ser_MRI_values_iter.index.intersection(ser_weights.index)  
                ser_weights = ser_weights[index_weights]
                ser_distribution = ser_MRI_values_iter.copy()
                ser_distribution = abs(ser_distribution - ser_distribution.iloc[-1])
                ser_distribution.sort_values(ascending = False, inplace = True)
                df_distribution = ser_distribution.to_frame('From_Last')
                df_distribution['Weights'] = ser_weights.values
                df_distribution.drop(columns = ['From_Last'], axis = 1, inplace = True)
                ser_distribution = df_distribution.squeeze()
                ser_weights = ser_distribution.sort_index()          
            ### Exponential volatility calculating:
            expvol_y = ser_returns_iter[index_iter]
            expvol_w = ser_weights[index_iter]             
            expvol_w = expvol_w / expvol_w.sum()
            expvol_results = np.sqrt(expvol_w.dot(expvol_y * expvol_y)) * np.sqrt(num_year_months)
            arr_results_container.append(expvol_results)      
            arr_index_container.append((str_country_iter, date_iter))
    ### Results index creating:
    index_wls_results = pd.MultiIndex.from_tuples(arr_index_container, names = ('Country', 'DatePoint'))
    ### Results aggregating:
    ser_expvol_results = pd.Series(arr_results_container, index = index_wls_results)
    
    print('hh_msci_expvol:', counter_points, 'MSCI monthly returns country/date points taken.')     
    print('hh_msci_expvol:', counter_regressions, 'MSCI on MRI correct intersections choosed')
    print('hh_msci_expvol: MSCI exponential volatility with weights distribution option "', weights_distribution, '" calculated successfully.')
    
    return ser_expvol_results


def hh_bokeh_MSCI_membership_map(path_countries_map_shp, df_expvol_all, df_country_codes, df_history_membership, figure_size, figure_title):
    """
    Version 0.01 2019-05-30
    
    FUNCTIONALITY:
      Drawing world map for illustrating MSCI membership evolution
    OUTPUT:
      bokeh.layouts.column that consists of:
          fig_world_map (bokeh.plotting.figure) - figure to display
          slider_dates (bokeh.models.widgets.DateSlider) - slider to date choosing for showing crossectional MSCI membership
    INPUT:
      path_countries_map_shp (string) - path to loacl file with world countries geo data
      df_expvol_all (pd.DataFrame) - resulats of MSCI on MRI exponential volatility
      df_country_codes (pd.DataFrame) - table of country codes
      df_history_membership (pd.DataFrame) - history of moving countries within classes     
      figure_size (tuple) - figure shapes
      figure_title (string) - figure title
    """

    import numpy as np
    import pandas as pd
    import geopandas as gpd
    from shapely.geometry import MultiPolygon, Polygon
    import bokeh.plotting as b_pl
    import bokeh.models as b_md    
    from bokeh.palettes import Set2, Set3
    from bokeh.layouts import widgetbox
    from bokeh.layouts import column as b_col
    from bokeh.models.widgets import DateSlider
    from dateutil.relativedelta import relativedelta
    from bokeh.models import CustomJS
    
    ### Integrating classes average to expvol data table
    df_history_membership.drop('Country', axis = 1, inplace = True)
    for member_code, (date_start, date_end, class_name) in df_history_membership.iterrows():
        index_iter_full = pd.date_range(start = date_start, end = date_end, freq = 'BM')
        if (index_iter_full.size > 0):
            if (df_expvol_all.index.isin([member_code], level = 0).sum() > 0):
                index_iter_returns = df_expvol_all.loc[member_code].index.intersection(index_iter_full)
                df_expvol_all.loc[(member_code, index_iter_returns), 'Class'] = class_name 
    df_expvol_all = df_expvol_all.reset_index()       
    df_expvol_all.set_index(['Class', 'Country', 'DatePoint'], inplace = True)
    df_expvol_class = df_expvol_all.copy()
    df_expvol_class = df_expvol_class.groupby(['Class', 'DatePoint']).mean()
    df_expvol_class = df_expvol_class.reset_index()
    df_expvol_class['Country'] = df_expvol_class['Class'] + ' - Average'
    df_expvol_class.set_index(['Class', 'Country', 'DatePoint'], inplace = True)
    df_expvol_all = pd.concat([df_expvol_all, df_expvol_class])
    df_expvol_all.sort_index(axis = 0, level = [0, 1], inplace = True)
    df_expvol_all = df_expvol_all.reset_index('DatePoint')
    ### Integrating long codes to expvol data table:
    df_expvol_long = df_expvol_all.reset_index()
    df_expvol_long = df_expvol_long.merge(df_country_codes, how = 'left', left_on = 'Country', right_on = 'ISO SHORT')
    df_expvol_long.drop(['ISO SHORT'], axis = 1, inplace = True)
    df_expvol_long.rename(columns = {'Country': 'Code_Short', 'ISO LONG': 'Code_Long'}, inplace = True)
    df_expvol_long['Code_Long'].fillna('CLASS', inplace = True)
    df_expvol_long.set_index(['DatePoint', 'Class', 'Code_Long', 'Code_Short'], inplace = True)
    df_expvol_long.sort_index(axis = 0, level = ['DatePoint', 'Class', 'Code_Long'], inplace = True)
    ### Exporting geo data:
    file_countries_gdf = gpd.read_file(path_countries_map_shp)[['ADMIN', 'ADM0_A3', 'geometry']]
    file_countries_gdf.columns = ['Country_Name', 'Country_Long_Code', 'Country_Geometry']
    file_countries_gdf.sort_values('Country_Long_Code', inplace = True)
    file_countries_gdf.drop(file_countries_gdf[file_countries_gdf['Country_Name'] == 'Antarctica'].index, axis = 0, inplace = True)
    ### Converting polygons to coordinates arrays dataframe for bokech patches:
    arr_names = []
    arr_codes = []
    arr_arr_x = []
    arr_arr_y = []
    for country_counter, (country_name, country_long_code, country_geometry) in file_countries_gdf.iterrows():
        if isinstance(country_geometry,  MultiPolygon):
            for country_polygon in country_geometry:
                arr_names.append(country_name)            
                arr_codes.append(country_long_code)
                arr_arr_x.append(list(country_polygon.exterior.coords.xy[0]))
                arr_arr_y.append(list(country_polygon.exterior.coords.xy[1]))             
        else:
            country_polygon = country_geometry
            arr_names.append(country_name)               
            arr_codes.append(country_long_code)
            arr_arr_x.append(list(country_polygon.exterior.coords.xy[0]))
            arr_arr_y.append(list(country_polygon.exterior.coords.xy[1]))     
    tup_country_coords = tuple(zip(arr_names, arr_codes, arr_arr_x, arr_arr_y))
    df_country_coords = pd.DataFrame(list(tup_country_coords), columns = ['Country_Name', 'Country_Long_Code', 'Coord_X_Array', 'Coord_Y_Array'])
    df_country_coords.set_index(['Country_Long_Code', 'Country_Name'], inplace = True)    
    ### Creating data table for class evolution through month/year to further connect with Slider bokeh widget:
    df_class_evolution = df_expvol_long.reset_index()
    df_class_evolution = df_class_evolution[['DatePoint', 'Class', 'Code_Long']]
    df_class_evolution['Year_Month'] = df_class_evolution['DatePoint'].dt.to_period('M')
    df_class_evolution.drop('DatePoint', axis = 1, inplace = True)
    df_class_evolution.set_index(['Year_Month'], inplace = True)
    ### Preparing dictionary of class membership for world map countries on each available year / month:
    dict_date_classes = {}
    for iter_index in df_class_evolution.index.unique():
        df_iter_class = df_class_evolution.loc[iter_index]
        df_iter_class = df_iter_class.reset_index()
        df_iter_class.set_index('Code_Long', inplace = True)
        df_iter_class.drop('Year_Month', axis = 1 ,inplace = True)
        df_iter_country = df_country_coords.reset_index(level = [1])['Country_Name'].to_frame().merge(df_iter_class, how = 'left', 
                                                                                                      left_on = 'Country_Long_Code', right_index = True)
        df_iter_country.drop('Country_Name', axis = 1, inplace = True)
        dict_date_classes[iter_index.strftime('%Y-%m')] = df_iter_country['Class'].values
    ### Configuring latest known MSCI classes to future world map data source:
    df_current_class = df_class_evolution.loc[df_class_evolution.index[-1]]
    df_current_class = df_current_class.reset_index()
    df_current_class.set_index('Code_Long', inplace = True)
    df_current_class.drop('Year_Month', axis = 1 ,inplace = True)
    df_country_coords_class = df_country_coords.merge(df_current_class, how = 'left', left_on = 'Country_Long_Code', right_index = True) 
    
    ### Defining output to notebook or to html file:
    b_pl.output_notebook()
    ### Data Source defining
    src_country_coords =  b_pl.ColumnDataSource(df_country_coords_class.reset_index())
    ### Colors categorising and mapping:
    arr_classes_name = list(df_country_coords_class['Class'].sort_values().dropna().unique())[::-1]
    arr_classes_num = list(list(np.arange(len(arr_classes_name)) + 0.5))
    arr_classes_str = list(map(str, arr_classes_num))
    dict_classes_label = dict(zip(arr_classes_str, arr_classes_name))
    categorical_cm_class = b_md.CategoricalColorMapper(factors = arr_classes_name, palette = Set2[4], nan_color = 'silver')
    linear_cm_class = b_md.LinearColorMapper(low = 0, high = len(arr_classes_name), palette = Set2[4], nan_color = 'silver')
    ### Initialising figure:
    str_fig_toolbar =  'pan, wheel_zoom, reset'
    tup_fig_size = figure_size
    str_fig_title = figure_title
    fig_world_map = b_pl.figure(tools = str_fig_toolbar, active_scroll = 'wheel_zoom', plot_width = tup_fig_size[0], plot_height = tup_fig_size[1], title = str_fig_title)
    ### Tuning figure:
    fig_world_map.axis.visible = False
    fig_world_map.xgrid.visible = False
    fig_world_map.ygrid.visible = False
    fig_world_map.add_tools(b_md.HoverTool(tooltips = [('Country', '@Country_Name')]))
    ### Drawing world map:
    fig_world_map.patches('Coord_X_Array', 'Coord_Y_Array', source = src_country_coords,
                          color = {'field': 'Class', 'transform': categorical_cm_class}, fill_alpha = 1.0, line_color = 'lightgray', line_width = 1.0)
    color_bar_class = b_md.ColorBar(color_mapper = linear_cm_class, label_standoff = 8, width = 20, height = tup_fig_size[1] - 100,
                                    border_line_color = None, orientation = 'vertical', location = (0, 50),
                                    ticker = b_md.FixedTicker(ticks = arr_classes_num), major_label_overrides = dict_classes_label,
                                    title = 'Class colors')
    fig_world_map.add_layout(color_bar_class, 'right')
    fig_world_map.toolbar.autohide = True    
    ### Drawing and tuning slider, creating a trigger for it's changes:
    callback_slider = CustomJS(args = dict(fig_to_update = fig_world_map, title_main_part = str_fig_title,
                                           cds_world = src_country_coords, arr_date_classes = dict_date_classes),
                               code = """
                                      var date_chosen = new Date(cb_obj.value);
                                      var arr_months = ['January', 'February', 'March', 'April', 'May', 'June', 
                                                        'July', 'August', 'September', 'October', 'November', 'December'];
                                      var date_month_name = arr_months[date_chosen.getMonth()]; 
                                      fig_to_update.title.text = title_main_part + ': ' + date_chosen.getFullYear() + ' / ' + date_month_name;
                                      var date_year_month = date_chosen.getFullYear().toString()
                                      if (date_chosen.getMonth() > 8)
                                      {
                                          date_year_month = date_year_month + '-' + (date_chosen.getMonth() + 1).toString();
                                      }
                                      else
                                      {
                                          date_year_month = date_year_month + '-' + '0' + (date_chosen.getMonth() + 1).toString();
                                      }
                                      cds_world.data['Class'] = arr_date_classes[date_year_month];
                                      cds_world.change.emit();                               
                                      fig_to_update.change.emit();
                                      """)
    date_min = df_expvol_long.index.get_level_values(level = 0).min()
    date_max = df_expvol_long.index.get_level_values(level = 0).max()
    slider_dates = DateSlider(title = 'Date to show classes membership', width = tup_fig_size[0] - 50, start = date_min, end = date_max, step = 1, value = date_max)
    slider_dates.js_on_change('value', callback_slider)
    slider_dates.callback_policy = 'throttle'
    slider_dates.tooltips = False
    ### Constructing common layout: 
    layout_world = b_col(fig_world_map, widgetbox(slider_dates))    

    return layout_world


def hh_bokeh_MSCI_MRI_expvol_map(path_countries_map_shp, df_expvol_all, df_country_codes, df_history_membership):
    """
    Version 0.01 2019-06-04
    
    FUNCTIONALITY:
      Drawing world map and synchronous plots for illustrating MSCI on MRI exponential volatility
    OUTPUT:
      bokeh.layouts.column that consists of:
      bokeh.layouts.column that consists of: 
          select_class (bokeh.models.widgets.inputs) - selection of class to show (including all classes)
          select_weights (bokeh.models.widgets.inputs) - selection the way of MSCI returns weighting to calculate expvol
      fig_world_map (bokeh.plotting.figure) - world map figure to display
      slider_dates (bokeh.models.widgets.DateSlider) - slider to date choosing for showing crossectional MSCI membership
      fig_expvol_plot (bokeh.plotting.figure) - expvol history plot figure to display
      select_country (bokeh.models.widgets.inputs) - selection country to draw additional expvol history plot
    INPUT:
      path_countries_map_shp (string) - path to local file with world countries geo data
      df_expvol_all (pd.DataFrame) - results of MSCI on MRI exponential volatility
      df_country_codes (pd.DataFrame) - table of country codes
      df_history_membership (pd.DataFrame) - history of moving countries within MSCI classes     
    """
    
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    from shapely.geometry import MultiPolygon, Polygon    
    import bokeh.plotting as b_pl
    import bokeh.models as b_md    
    from bokeh.palettes import Set2, Set3, RdYlGn, YlOrRd, inferno
    from bokeh.layouts import widgetbox
    from bokeh.layouts import column as b_col
    from bokeh.layouts import row as b_row
    from bokeh.models.widgets import DateSlider, Select
    from bokeh.models import CustomJS
    
    ### Integrating classes average and common average to expvol data table:
    for member_code, (date_start, date_end, class_name) in df_history_membership.iterrows():
        index_iter_full = pd.date_range(start = date_start, end = date_end, freq = 'BM')
        if (index_iter_full.size > 0):
            if (df_expvol_all.index.isin([member_code], level = 0).sum() > 0):
                index_iter_returns = df_expvol_all.loc[member_code].index.intersection(index_iter_full)
                df_expvol_all.loc[(member_code, index_iter_returns), 'Class'] = class_name 
    df_expvol_all = df_expvol_all.reset_index()       
    df_expvol_all.set_index(['Class', 'Country', 'DatePoint'], inplace = True)
    ### Constructing class average:
    df_expvol_class = df_expvol_all.copy()
    df_expvol_class = df_expvol_class.groupby(['Class', 'DatePoint']).mean()
    df_expvol_class = df_expvol_class.reset_index()
    df_expvol_class['Country'] = df_expvol_class['Class'] + ' - Average'
    df_expvol_class.set_index(['Class', 'Country', 'DatePoint'], inplace = True)
    ### Constructing common average:
    df_expvol_common = df_expvol_all.copy()
    df_expvol_common = df_expvol_common.groupby(['DatePoint']).mean()
    df_expvol_common = df_expvol_common.reset_index()
    df_expvol_common = df_expvol_common.assign(Country = 'ALL - Average', Class = 'ALL')
    df_expvol_common.set_index(['Class', 'Country', 'DatePoint'], inplace = True)
    ### Consolidating extended expvol data table:
    df_expvol_all = pd.concat([df_expvol_all, df_expvol_class])
    df_expvol_all = pd.concat([df_expvol_all, df_expvol_common])
    df_expvol_all.sort_index(axis = 0, level = [0, 1], inplace = True)
    df_expvol_all = df_expvol_all.reset_index('DatePoint')
    ### Integrating long codes to expvol data table:
    df_expvol_long = df_expvol_all.reset_index()
    df_expvol_long = df_expvol_long.merge(df_country_codes, how = 'left', left_on = 'Country', right_on = 'ISO SHORT')
    df_expvol_long.drop(['ISO SHORT'], axis = 1, inplace = True)
    df_expvol_long.rename(columns = {'Country': 'Code_Short', 'ISO LONG': 'Code_Long'}, inplace = True)
    df_expvol_long['Code_Long'].fillna('CLASS', inplace = True)
    df_expvol_long.set_index(['DatePoint', 'Class', 'Code_Long', 'Code_Short'], inplace = True)
    df_expvol_long.sort_index(axis = 0, level = ['DatePoint', 'Class', 'Code_Long'], inplace = True)
    ### Exporting geo data:
    import geopandas as gpd
    file_countries_gdf = gpd.read_file(path_countries_map_shp)[['ADMIN', 'ADM0_A3', 'geometry']]
    file_countries_gdf.columns = ['Country_Name', 'Country_Long_Code', 'Country_Geometry']
    file_countries_gdf.sort_values('Country_Long_Code', inplace = True)
    file_countries_gdf.drop(file_countries_gdf[file_countries_gdf['Country_Name'] == 'Antarctica'].index, axis = 0, inplace = True)
    ### Converting polygons to coordinates arrays dataframe for bokech patches:
    from shapely.geometry import MultiPolygon, Polygon
    arr_names = []
    arr_codes = []
    arr_arr_x = []
    arr_arr_y = []
    ### Extacting polygons from gdf file:
    for country_counter, (country_name, country_long_code, country_geometry) in file_countries_gdf.iterrows():
        ### For group of polygons:
        if isinstance(country_geometry,  MultiPolygon):
            for country_polygon in country_geometry:
                arr_names.append(country_name)            
                arr_codes.append(country_long_code)
                arr_arr_x.append(list(country_polygon.exterior.coords.xy[0]))
                arr_arr_y.append(list(country_polygon.exterior.coords.xy[1]))             
        ### For single polygons:            
        else:
            country_polygon = country_geometry
            arr_names.append(country_name)               
            arr_codes.append(country_long_code)
            arr_arr_x.append(list(country_polygon.exterior.coords.xy[0]))
            arr_arr_y.append(list(country_polygon.exterior.coords.xy[1]))  
    ### Constructing dataframe with row for each polygon:            
    tup_country_coords = tuple(zip(arr_names, arr_codes, arr_arr_x, arr_arr_y))
    df_country_coords = pd.DataFrame(list(tup_country_coords), columns = ['Country_Name', 'Country_Long_Code', 'Coord_X_Array', 'Coord_Y_Array'])
    df_country_coords.set_index(['Country_Long_Code', 'Country_Name'], inplace = True)
    ### Creating data table for class evolution through month/year to further connect with Slider bokeh widget:
    df_class_evolution = df_expvol_long.reset_index()
    df_class_evolution = df_class_evolution[['DatePoint', 'Class', 'Code_Long', 'None', 'By_Date', 'By_Value']]
    df_class_evolution['Year_Month'] = df_class_evolution['DatePoint'].dt.to_period('M')
    df_class_evolution.drop('DatePoint', axis = 1, inplace = True)
    df_class_evolution.set_index(['Year_Month'], inplace = True)
    ### Preparing dictionary of class membership for world map countries on each available month/year:
    arr_classes_name = np.append(df_class_evolution['Class'].unique(), 'ALL')
    arr_weights_name = np.array(['None', 'By_Date', 'By_Value'])
    dict_date_classes = {}
    dict_expvol_general = {}
    for iter_class_name in arr_classes_name:
        dict_expvol_general[iter_class_name] = {}
        for iter_weight_name in arr_weights_name:
            dict_expvol_general[iter_class_name][iter_weight_name] = {}    
    dict_iter_country = {}
    for iter_index in df_class_evolution.index.unique():
        df_iter_class_plus = df_class_evolution.loc[iter_index]
        df_iter_class_plus = df_iter_class_plus.reset_index()
        df_iter_class_plus.set_index('Code_Long', inplace = True)
        df_iter_class_plus.drop('Year_Month', axis = 1 ,inplace = True)
        df_iter_country = df_country_coords.reset_index(level = [1])['Country_Name'].to_frame().merge(df_iter_class_plus, how = 'left', 
                                                                                                      left_on = 'Country_Long_Code', right_index = True)
        df_iter_country.drop('Country_Name', axis = 1, inplace = True)
        dict_date_classes[iter_index.strftime('%Y-%m')] = df_iter_country['Class'].values  
        for iter_class_name in arr_classes_name:
            df_iter_country_class = df_iter_country.copy()
            if (iter_class_name != 'ALL'):
                df_iter_country_class.loc[df_iter_country[df_iter_country['Class'] != iter_class_name].index, ['None', 'By_Date', 'By_Value']] = np.NaN           
            dict_iter_country[iter_class_name] = df_iter_country_class
            for iter_weight_name in arr_weights_name:
                dict_expvol_general[iter_class_name][iter_weight_name][iter_index.strftime('%Y-%m')] = dict_iter_country[iter_class_name][iter_weight_name].values
    ### Configuring latest known MSCI data as initial world map classification data source:
    df_current_date = df_class_evolution.loc[df_class_evolution.index[-1]]
    df_current_date = df_current_date.reset_index()
    df_current_date.set_index('Code_Long', inplace = True)
    df_current_date.drop('Year_Month', axis = 1 ,inplace = True)
    df_country_coords_class = df_country_coords.merge(df_current_date, how = 'left', left_on = 'Country_Long_Code', right_index = True)
    df_country_coords_class['Active'] = df_country_coords_class['None']
    ### Configuring actual MSCI common average as initial class average expvol plot data source:
    df_class_average = df_class_evolution[df_class_evolution['Class'] == 'ALL'].copy()
    df_class_average['Active'] = df_class_average['None']
    ### Creating list for countrty selection:
    df_expvol_countries = df_class_evolution.reset_index().merge(df_country_codes.reset_index(), how = 'left', left_on = 'Code_Long', right_on = 'ISO LONG')
    df_countries_list = df_expvol_countries.drop_duplicates(subset = ['COUNTRY']).dropna().reset_index()
    df_countries_list.sort_values('COUNTRY', inplace = True)
    ### Reconfiguring membership history for long ISO codes
    df_history_membership_long = df_history_membership.merge(df_country_codes, how = 'left', left_index = True, right_on = 'ISO SHORT').reset_index()
    df_history_membership_long.head()
    df_history_membership_long.drop(['COUNTRY', 'ISO SHORT'], axis = 1, inplace = True)
    df_history_membership_long.rename(columns = {'ISO LONG': 'Member Code'}, inplace = True)
    df_history_membership_long.set_index('Member Code')
    ### Creating transit dataframe as container for data transfer betwwen widgets and figures:
    df_meta_transit = pd.DataFrame([str(df_class_evolution.index[-1]), 'ALL', 'None', 'None'], index = ['Year_Month', 'Class', 'Weights', 'Country'])  
    
    ### Defining output to notebook or to html file:
    b_pl.output_notebook()
    ### Bokeh Data Source defining:
    src_meta_transit =  b_pl.ColumnDataSource(df_meta_transit.T) ### For data transfer betwwen elements
    src_country_coords =  b_pl.ColumnDataSource(df_country_coords_class.reset_index()) ### For world map borders
    src_expvol_evolution = b_pl.ColumnDataSource(df_class_evolution.reset_index()) ### For plotting country level expvol values (common data)
    src_class_average = b_pl.ColumnDataSource(df_class_average.reset_index()) ### For plotting class average expvol values
    src_country_chosen = b_pl.ColumnDataSource(df_class_average.copy().reset_index()) ### For plotting country level expvol values (chosen country)
    src_class_boxes = b_md.ColumnDataSource(df_history_membership_long) ### For background color box annotations
    ### Plot boundary dates defining:
    fig_min_date = df_class_average.reset_index()['Year_Month'].min().to_timestamp()
    fig_max_date = df_class_average.reset_index()['Year_Month'].max().to_timestamp()
    ### Weights select additional data connectors constructing:
    dict_weights_select = {'Priority: none (equal weights)': 'None', 'Priority: by date': 'By_Date', 'Priority: by similarity to current value (MRI vector)': 'By_Value'}
    dict_weights_mirror = dict(zip(dict_weights_select.values(), dict_weights_select.keys()))
    arr_weights_select = list(dict_weights_select.keys())
    ### Class select additional data connectors constructing:
    dict_classes_select = {'Developed Markets': 'DM', 'Emerging Markets': 'EM', 'Frontier Markets': 'FM', 'Standalone Markets': 'SM', 'All Markets': 'ALL'}
    dict_classes_mirror = dict(zip(dict_classes_select.values(), dict_classes_select.keys()))
    arr_classes_select = list(dict_classes_select.keys())
    ### Country select additional data connectors constructing:    
    arr_country_select_list = list(np.append((df_countries_list['COUNTRY'] + ' (' + df_countries_list['Code_Long'] + ')').values, 'No country selected'))
    arr_country_code_list = list(np.append(df_countries_list['Code_Long'].values, 'None'))
    dict_country_list = dict(zip(arr_country_select_list, arr_country_code_list))
    dict_country_mirror = dict(zip(arr_country_code_list, arr_country_select_list))
    ### Colors categorising and mapping:
    linear_cm_expvol = b_md.LinearColorMapper(low = 0, high = 1, palette = inferno(20)[::-1], nan_color = 'silver')
    arr_classes_name = list(dict_classes_select.values())[: -1][::-1]
    arr_classes_num = list(list(np.arange(len(arr_classes_name)) + 0.5))
    arr_classes_str = list(map(str, arr_classes_num))
    dict_classes_label = dict(zip(arr_classes_str, arr_classes_name))
    dict_classes_colors = dict(zip(arr_classes_name, Set2[4]))
    categorical_cm_class = b_md.CategoricalColorMapper(factors = arr_classes_name, palette = Set2[4], nan_color = 'silver')
    linear_cm_class = b_md.LinearColorMapper(low = 0, high = len(arr_classes_name), palette = Set2[4], nan_color = 'silver')
    ### Initialising expvol values plot figure:
    str_fig_expvol_toolbar =  'pan, wheel_zoom, reset'
    tup_fig_expvol_size = (800, 200)
    str_fig_expvol_title = 'Exponential volatility: '
    fig_expvol_plot = b_pl.figure(tools = str_fig_expvol_toolbar, active_scroll = 'wheel_zoom', plot_width = tup_fig_expvol_size[0], plot_height = tup_fig_expvol_size[1],
                                  title = str_fig_expvol_title + dict_classes_mirror['ALL'] + ' / ' + dict_weights_mirror['None'],
                                  x_axis_type = 'datetime', x_range = (fig_min_date, fig_max_date)) 
    ### Drawing expvol plots:
    line_class_plot = fig_expvol_plot.line(x = 'Year_Month', y = 'Active', source = src_class_average,
                                           line_color = 'blue', line_width = 2.0, line_dash = 'dotted')
    line_country_plot = fig_expvol_plot.line(x = 'Year_Month', y = 'Active', source = src_country_chosen,
                                             line_color = 'red', line_width = 2.0, line_dash = 'dotted', line_alpha = 0.0)
    ### Adding classes color bar to expvol plot figure:
    color_bar_class = b_md.ColorBar(color_mapper = linear_cm_class, scale_alpha = 0.2, label_standoff = 8, width = 20, height = tup_fig_expvol_size[1] - 50,
                                    border_line_color = None, orientation = 'vertical', location = (0, -10),
                                    ticker = b_md.FixedTicker(ticks = arr_classes_num), major_label_overrides = dict_classes_label,
                                    title = 'Class')
    fig_expvol_plot.add_layout(color_bar_class, 'right')
    ### Adding current data marker to expvol plot figure:    
    span_current_date = b_md.Span(location = fig_max_date, dimension = 'height', line_color = 'gray', line_dash = 'dashed', line_width = 3.0)
    fig_expvol_plot.add_layout(span_current_date)
    ### Adding dynamic legend to expvol plot figure:     
    legend_expvol_plot = b_md.Legend(items = [('Class Average', [line_class_plot]), (arr_country_select_list[-1] , [line_country_plot])], location = (0, 0))
    fig_expvol_plot.add_layout(legend_expvol_plot)
    ### Adding array of blank annotations (class evolution stages) to expvol plot figure:
    num_annotations = 5
    arr_annotations = []
    for counter_annotation in np.arange(num_annotations):
        arr_annotations.append(b_md.BoxAnnotation(left = fig_max_date, right = fig_max_date, fill_alpha = 0.2, fill_color = 'white'))
        fig_expvol_plot.add_layout(arr_annotations[counter_annotation])
    ### Tuning expvol plot figure:
    fig_expvol_plot.legend.location = 'top_left'
    fig_expvol_plot.legend.background_fill_alpha  = 0.75
    fig_expvol_plot.yaxis.formatter = b_md.NumeralTickFormatter(format = '.00')
    ### Creating hover inspection to expvol plot figure:
    fig_expvol_plot.add_tools(b_md.HoverTool(tooltips = [('Country code', '@Code_Long'), ('Month', '@Year_Month{%Y-%m}'), ('ExpVol (equal weights)', '@None{0,0.000}'),
                                                         ('ExpVol (by date weights)', '@By_Date{0,0.000}'), ('ExpVol (similarity weights)', '@By_Value{0,0.000}')], 
                                             formatters = {'None': 'numeral', 'By_Date': 'numeral', 'By_Value': 'numeral', 'Year_Month': 'datetime'}))
    ### Constructing select widget for choosing country tuning (including connection between selected value and plot):
    callback_country_select = CustomJS(args = dict(fig_plot_to_update = fig_expvol_plot, legend_expvol = legend_expvol_plot, line_country = line_country_plot,
                                                   arr_country_codes = dict_country_list,
                                                   cds_transit = src_meta_transit, cds_country_plot = src_country_chosen, cds_full = src_expvol_evolution,
                                                   cds_annotations = src_class_boxes, date_min = fig_min_date, date_max = fig_max_date,
                                                   arr_annotations = arr_annotations, num_annotations = num_annotations, dict_colors = dict_classes_colors),
                                       code = """                                  
                                              var country_chosen = cb_obj.value;
                                              var country_code = arr_country_codes[country_chosen];
                                              cds_transit.data['Country'] = country_code;
                                              var weights_selected = cds_transit.data['Weights'];                                
                                              if (country_code == 'None')
                                              {
                                                  line_country.glyph.line_alpha = 0.0;                                             
                                              }
                                              else
                                              {
                                                  line_country.glyph.line_alpha = 1.0;                                           
                                              }
                                              var legend_expvol_last = (legend_expvol.items.length - 1);
                                              legend_expvol.items[legend_expvol_last].label['value'] = country_chosen;
                                              legend_expvol.change.emit();  
                                              cds_country_plot.data['Year_Month'] = [];
                                              cds_country_plot.data['Class'] = [];
                                              cds_country_plot.data['Code_Long'] = [];
                                              cds_country_plot.data['None'] = [];
                                              cds_country_plot.data['By_Date'] = [];
                                              cds_country_plot.data['By_Value'] = [];
                                              cds_country_plot.data['Active'] = []; 
                                              for (var iter_counter = 0; iter_counter <= cds_full.get_length(); iter_counter++)
                                              {
                                                  if ((cds_full.data['Code_Long'][iter_counter] == country_code))
                                                  {
                                                      cds_country_plot.data['Year_Month'].push(cds_full.data['Year_Month'][iter_counter]);
                                                      cds_country_plot.data['Class'].push(cds_full.data['Class'][iter_counter]);
                                                      cds_country_plot.data['Code_Long'].push(cds_full.data['Code_Long'][iter_counter]);
                                                      cds_country_plot.data['None'].push(cds_full.data['None'][iter_counter]);
                                                      cds_country_plot.data['By_Date'].push(cds_full.data['By_Date'][iter_counter]);
                                                      cds_country_plot.data['By_Value'].push(cds_full.data['By_Value'][iter_counter]);
                                                      cds_country_plot.data['Active'].push(cds_full.data[weights_selected][iter_counter]);
                                                  }
                                              }                                          
                                              cds_country_plot.change.emit();
                                              var iter_ann;
                                              for (iter_ann = 0; iter_ann < num_annotations; iter_ann++)
                                              {
                                                arr_annotations[iter_ann].fill_color = 'white';
                                                arr_annotations[iter_ann].left = date_max;
                                                arr_annotations[iter_ann].right = date_max;
                                              }
                                              iter_ann = 0;
                                              var iter_history;
                                              for (iter_history = 0; iter_history <= cds_annotations.get_length(); iter_history++)
                                              {
                                                if (cds_annotations.data['Member Code'][iter_history] == country_code)
                                                {
                                                    arr_annotations[iter_ann].fill_color = dict_colors[cds_annotations.data['Index Name'][iter_history]];
                                                    arr_annotations[iter_ann].left = cds_annotations.data['Start Date'][iter_history];
                                                    arr_annotations[iter_ann].right = cds_annotations.data['End Date'][iter_history];
                                                    iter_ann = iter_ann + 1;
                                                }
                                              }                                                  
                                              fig_plot_to_update.change.emit(); 
                                              """)
    select_country = Select(title = 'Select country to show plot:', options = arr_country_select_list, value = arr_country_select_list[-1], callback = callback_country_select)
    ### Initialising worldmap classes evolution figure:
    str_fig_worldmap_toolbar =  'pan, wheel_zoom, reset'
    tup_fig_worldmap_size = (tup_fig_expvol_size[0], tup_fig_expvol_size[1] * 2)
    str_fig_worldmap_title = 'MSCI Membership exponential volatility graduation'
    fig_world_map = b_pl.figure(tools = str_fig_worldmap_toolbar, active_scroll = 'wheel_zoom', plot_width = tup_fig_worldmap_size[0], plot_height = tup_fig_worldmap_size[1],
                                title = str_fig_worldmap_title)
    ### Tuning worldmap figure:
    fig_world_map.axis.visible = False
    fig_world_map.xgrid.visible = False
    fig_world_map.ygrid.visible = False
    fig_world_map.toolbar.autohide = True
    ### Creating hover inspection to worldmap figure:    
    fig_world_map.add_tools(b_md.HoverTool(tooltips = [('Country', '@Country_Name'), ('MSCI Class', '@Class'), ('ExpVol (equal weights)', '@None{0,0.000}'),
                                                       ('ExpVol (by date weights)', '@By_Date{0,0.000}'), ('ExpVol (similarity weights)', '@By_Value{0,0.000}')], 
                                           formatters = {'None': 'numeral', 'By_Date': 'numeral', 'By_Value': 'numeral'}))
    ### Drawing world map:
    patches_world_map = fig_world_map.patches('Coord_X_Array', 'Coord_Y_Array', source = src_country_coords,
                                              color = {'field': 'Active', 'transform': linear_cm_expvol}, fill_alpha = 1.0, line_color = 'lightgray', 
                                              line_width = 1.0)
    ### Adding expvol value color bar to expvol plot figure: 
    color_bar_class = b_md.ColorBar(color_mapper = linear_cm_expvol, label_standoff = 8, width = 20, height = tup_fig_worldmap_size[1] - 100,
                                    border_line_color = None, orientation = 'vertical', location = (0, 50),
                                    ticker = b_md.AdaptiveTicker(desired_num_ticks = len(linear_cm_expvol.palette)),
                                    formatter = b_md.NumeralTickFormatter(format = '.00'))
    fig_world_map.add_layout(color_bar_class, 'right')
    ### Drawing and tuning date slider widget, creating a trigger for it's changes:
    callback_date_slider = CustomJS(args = dict(fig_map_to_update = fig_world_map, title_map_main_part = str_fig_worldmap_title, 
                                                fig_plot_to_update = fig_expvol_plot, span_plot = span_current_date,
                                                cds_world = src_country_coords, cds_transit = src_meta_transit, cds_plot = src_expvol_evolution,
                                                arr_date_classes = dict_date_classes, arr_date_expvol_general = dict_expvol_general),
                                    code = """                             
                                           var date_chosen = new Date(cb_obj.value);
                                           var class_selected = cds_transit.data['Class'];
                                           var weights_selected = cds_transit.data['Weights'];
                                           var arr_months = ['January', 'February', 'March', 'April', 'May', 'June', 
                                                             'July', 'August', 'September', 'October', 'November', 'December'];
                                           var date_month_name = arr_months[date_chosen.getMonth()]; 
                                           fig_map_to_update.title.text = title_map_main_part + ': ' + date_chosen.getFullYear() + ' / ' + date_month_name;
                                           var date_year_month = date_chosen.getFullYear().toString();
                                           if (date_chosen.getMonth() > 8)
                                           {
                                               date_year_month = date_year_month + '-' + (date_chosen.getMonth() + 1).toString();
                                           }
                                           else
                                           {
                                               date_year_month = date_year_month + '-' + '0' + (date_chosen.getMonth() + 1).toString();
                                           }                                  
                                           cds_transit.data['Year_Month'] = date_year_month;                                       
                                           cds_world.data['Class'] = arr_date_classes[date_year_month];
                                           cds_world.data['None'] = arr_date_expvol_general[class_selected]['None'][date_year_month];
                                           cds_world.data['By_Date'] = arr_date_expvol_general[class_selected]['By_Date'][date_year_month];
                                           cds_world.data['By_Value'] = arr_date_expvol_general[class_selected]['By_Value'][date_year_month];                         
                                           cds_world.data['Active'] = cds_world.data[weights_selected];
                                           cds_world.change.emit();                               
                                           cds_transit.change.emit(); 
                                           fig_map_to_update.change.emit();           
                                           span_plot.location = cb_obj.value;
                                           span_plot.change.emit();
                                           fig_plot_to_update.change.emit();                                       
                                           """)
    date_min = df_expvol_long.index.get_level_values(level = 0).min()
    date_max = df_expvol_long.index.get_level_values(level = 0).max()
    slider_dates = DateSlider(title = 'Date to show classes membership', width = tup_fig_worldmap_size[0] - 50, start = date_min, end = date_max, step = 1, value = date_max)
    slider_dates.callback_policy = 'throttle'
    slider_dates.js_on_change('value', callback_date_slider)
    slider_dates.tooltips = False
    ### Creating select widget for choosing class tuning (including connection between selected value and range slider boundaries, figure plots, figure boundaries, figure title):
    callback_class_select = CustomJS(args = dict(fig_map_to_update = fig_world_map, fig_plot_to_update = fig_expvol_plot, title_plot_main_part = str_fig_expvol_title,
                                                 cds_world = src_country_coords, cds_transit = src_meta_transit, 
                                                 cds_full = src_expvol_evolution, cds_class_plot = src_class_average,
                                                 arr_classes_names = dict_classes_select, arr_date_expvol_general = dict_expvol_general,
                                                 arr_classes_mirror = dict_classes_mirror, arr_weights_mirror = dict_weights_mirror),
                                     code = """
                                            var date_year_month = cds_transit.data['Year_Month'];
                                            var class_selected = arr_classes_names[cb_obj.value];
                                            var weights_selected = cds_transit.data['Weights'];                                        
                                            cds_world.data['None'] = arr_date_expvol_general[class_selected]['None'][date_year_month];
                                            cds_world.data['By_Date'] = arr_date_expvol_general[class_selected]['By_Date'][date_year_month];
                                            cds_world.data['By_Value'] = arr_date_expvol_general[class_selected]['By_Value'][date_year_month];                         
                                            cds_world.data['Active'] = cds_world.data[weights_selected];     
                                            cds_transit.data['Class'] = class_selected;                                           
                                            cds_world.change.emit();                               
                                            cds_transit.change.emit(); 
                                            fig_map_to_update.change.emit();
                                            fig_plot_to_update.title.text = title_plot_main_part + arr_classes_mirror[class_selected] + ' / '
                                            fig_plot_to_update.title.text = fig_plot_to_update.title.text + arr_weights_mirror[weights_selected];
                                            cds_class_plot.data['Year_Month'] = [];
                                            cds_class_plot.data['Class'] = [];
                                            cds_class_plot.data['Code_Long'] = [];
                                            cds_class_plot.data['None'] = [];
                                            cds_class_plot.data['By_Date'] = [];
                                            cds_class_plot.data['By_Value'] = [];
                                            cds_class_plot.data['Active'] = [];                                        
                                            for (var iter_counter = 0; iter_counter <= cds_full.get_length(); iter_counter++)
                                            {
                                                if ((cds_full.data['Class'][iter_counter] == class_selected) && (cds_full.data['Code_Long'][iter_counter] == 'CLASS'))
                                                {
                                                    cds_class_plot.data['Year_Month'].push(cds_full.data['Year_Month'][iter_counter]);
                                                    cds_class_plot.data['Class'].push(cds_full.data['Class'][iter_counter]);
                                                    cds_class_plot.data['Code_Long'].push(cds_full.data['Code_Long'][iter_counter]);
                                                    cds_class_plot.data['None'].push(cds_full.data['None'][iter_counter]);
                                                    cds_class_plot.data['By_Date'].push(cds_full.data['By_Date'][iter_counter]);
                                                    cds_class_plot.data['By_Value'].push(cds_full.data['By_Value'][iter_counter]);
                                                    cds_class_plot.data['Active'].push(cds_full.data[weights_selected][iter_counter]);
                                                }
                                            }
                                            cds_class_plot.change.emit();                                     
                                            fig_plot_to_update.change.emit();
                                            """)
    select_class = Select(title = 'Select MSCI class to show:', options = arr_classes_select, value = arr_classes_select[-1], callback = callback_class_select)
    ### Creating select widget for choosing country (including connection between selected value and range slider boundaries, figure plots, figure boundaries, figure title):
    callback_weights_select = CustomJS(args = dict(fig_map_to_update = fig_world_map, fig_plot_to_update = fig_expvol_plot, title_plot_main_part = str_fig_expvol_title, 
                                                   cds_world = src_country_coords, cds_transit = src_meta_transit, 
                                                   cds_full = src_expvol_evolution, cds_class_plot = src_class_average, cds_country_plot = src_country_chosen,
                                                   arr_weights_names = dict_weights_select, arr_date_expvol_general = dict_expvol_general,
                                                   arr_classes_mirror = dict_classes_mirror, arr_weights_mirror = dict_weights_mirror),
                                       code = """
                                              var date_year_month = cds_transit.data['Year_Month'];
                                              var class_selected = cds_transit.data['Class'];
                                              var country_code = cds_transit.data['Country'];                                          
                                              var weights_selected = arr_weights_names[cb_obj.value];
                                              cds_world.data['Active'] = cds_world.data[weights_selected];
                                              cds_transit.data['Weights'] = weights_selected;
                                              cds_transit.change.emit();
                                              cds_world.change.emit();                                       
                                              fig_map_to_update.change.emit();
                                              fig_plot_to_update.title.text = title_plot_main_part + arr_classes_mirror[class_selected] + ' / '
                                              fig_plot_to_update.title.text = fig_plot_to_update.title.text + arr_weights_mirror[weights_selected];
                                              cds_class_plot.data['Year_Month'] = [];
                                              cds_class_plot.data['Class'] = [];
                                              cds_class_plot.data['Code_Long'] = [];
                                              cds_class_plot.data['None'] = [];
                                              cds_class_plot.data['By_Date'] = [];
                                              cds_class_plot.data['By_Value'] = [];
                                              cds_class_plot.data['Active'] = [];                                        
                                              for (var iter_counter = 0; iter_counter <= cds_full.get_length(); iter_counter++)
                                              {
                                                  if ((cds_full.data['Class'][iter_counter] == class_selected) && (cds_full.data['Code_Long'][iter_counter] == 'CLASS'))
                                                  {
                                                      cds_class_plot.data['Year_Month'].push(cds_full.data['Year_Month'][iter_counter]);
                                                      cds_class_plot.data['Class'].push(cds_full.data['Class'][iter_counter]);
                                                      cds_class_plot.data['Code_Long'].push(cds_full.data['Code_Long'][iter_counter]);
                                                      cds_class_plot.data['None'].push(cds_full.data['None'][iter_counter]);
                                                      cds_class_plot.data['By_Date'].push(cds_full.data['By_Date'][iter_counter]);
                                                      cds_class_plot.data['By_Value'].push(cds_full.data['By_Value'][iter_counter]);
                                                      cds_class_plot.data['Active'].push(cds_full.data[weights_selected][iter_counter]);
                                                  }
                                              }
                                              cds_class_plot.change.emit();
                                              cds_country_plot.data['Year_Month'] = [];
                                              cds_country_plot.data['Class'] = [];
                                              cds_country_plot.data['Code_Long'] = [];
                                              cds_country_plot.data['None'] = [];
                                              cds_country_plot.data['By_Date'] = [];
                                              cds_country_plot.data['By_Value'] = [];
                                              cds_country_plot.data['Active'] = []; 
                                              for (var iter_counter = 0; iter_counter <= cds_full.get_length(); iter_counter++)
                                              {
                                                  if ((cds_full.data['Code_Long'][iter_counter] == country_code))
                                                  {
                                                      cds_country_plot.data['Year_Month'].push(cds_full.data['Year_Month'][iter_counter]);
                                                      cds_country_plot.data['Class'].push(cds_full.data['Class'][iter_counter]);
                                                      cds_country_plot.data['Code_Long'].push(cds_full.data['Code_Long'][iter_counter]);
                                                      cds_country_plot.data['None'].push(cds_full.data['None'][iter_counter]);
                                                      cds_country_plot.data['By_Date'].push(cds_full.data['By_Date'][iter_counter]);
                                                      cds_country_plot.data['By_Value'].push(cds_full.data['By_Value'][iter_counter]);
                                                      cds_country_plot.data['Active'].push(cds_full.data[weights_selected][iter_counter]);
                                                  }
                                              }                                          
                                              cds_country_plot.change.emit();  
                                              fig_plot_to_update.change.emit();                                          
                                              """)
    select_weights = Select(title = 'Select way of weights allocation:', options = arr_weights_select, value = arr_weights_select[0], callback = callback_weights_select)
    ### Constructing common layout: 
    layout_world = b_col(b_row(widgetbox(select_class), widgetbox(select_weights)), fig_world_map, widgetbox(slider_dates), fig_expvol_plot, widgetbox(select_country))

    return layout_world


def hh_bokeh_MSCI_MRI_beta_map(path_countries_map_shp, df_beta_all, df_country_codes, df_history_membership):
    """
    Version 0.01 2019-06-04
    
    FUNCTIONALITY:
      Drawing world map and synchronous plots for illustrating MSCI on MRI betas
    OUTPUT:
      bokeh.layouts.column that consists of:
      bokeh.layouts.column that consists of: 
          select_class (bokeh.models.widgets.inputs) - selection of class to show (including all classes)
          select_weights (bokeh.models.widgets.inputs) - selection the way of MSCI returns weighting to calculate beta
      fig_world_map (bokeh.plotting.figure) - world map figure to display
      slider_dates (bokeh.models.widgets.DateSlider) - slider to date choosing for showing crossectional MSCI membership
      fig_beta_plot (bokeh.plotting.figure) - beta history plot figure to display
      select_country (bokeh.models.widgets.inputs) - selection country to draw additional beta history plot
    INPUT:
      path_countries_map_shp (string) - path to local file with world countries geo data
      df_beta_all (pd.DataFrame) - results of MSCI on MRI betas
      df_country_codes (pd.DataFrame) - table of country codes
      df_history_membership (pd.DataFrame) - history of moving countries within MSCI classes     
    """
    
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    from shapely.geometry import MultiPolygon, Polygon    
    import bokeh.plotting as b_pl
    import bokeh.models as b_md    
    from bokeh.palettes import Set2, Set3, RdYlGn, YlOrRd, inferno
    from bokeh.layouts import widgetbox
    from bokeh.layouts import column as b_col
    from bokeh.layouts import row as b_row
    from bokeh.models.widgets import DateSlider, Select
    from bokeh.models import CustomJS
    
    ### Integrating classes average and common average to beta data table:
    for member_code, (date_start, date_end, class_name) in df_history_membership.iterrows():
        index_iter_full = pd.date_range(start = date_start, end = date_end, freq = 'BM')
        if (index_iter_full.size > 0):
            if (df_beta_all.index.isin([member_code], level = 0).sum() > 0):
                index_iter_returns = df_beta_all.loc[member_code].index.intersection(index_iter_full)
                df_beta_all.loc[(member_code, index_iter_returns), 'Class'] = class_name 
    df_beta_all = df_beta_all.reset_index()       
    df_beta_all.set_index(['Class', 'Country', 'DatePoint'], inplace = True)
    ### Constructing class average:
    df_beta_class = df_beta_all.copy()
    df_beta_class = df_beta_class.groupby(['Class', 'DatePoint']).mean()
    df_beta_class = df_beta_class.reset_index()
    df_beta_class['Country'] = df_beta_class['Class'] + ' - Average'
    df_beta_class.set_index(['Class', 'Country', 'DatePoint'], inplace = True)
    ### Constructing common average:
    df_beta_common = df_beta_all.copy()
    df_beta_common = df_beta_common.groupby(['DatePoint']).mean()
    df_beta_common = df_beta_common.reset_index()
    df_beta_common = df_beta_common.assign(Country = 'ALL - Average', Class = 'ALL')
    df_beta_common.set_index(['Class', 'Country', 'DatePoint'], inplace = True)
    ### Consolidating extended beta data table:
    df_beta_all = pd.concat([df_beta_all, df_beta_class])
    df_beta_all = pd.concat([df_beta_all, df_beta_common])
    df_beta_all.sort_index(axis = 0, level = [0, 1], inplace = True)
    df_beta_all = df_beta_all.reset_index('DatePoint')
    ### Integrating long codes to beta data table:
    df_beta_long = df_beta_all.reset_index()
    df_beta_long = df_beta_long.merge(df_country_codes, how = 'left', left_on = 'Country', right_on = 'ISO SHORT')
    df_beta_long.drop(['ISO SHORT'], axis = 1, inplace = True)
    df_beta_long.rename(columns = {'Country': 'Code_Short', 'ISO LONG': 'Code_Long'}, inplace = True)
    df_beta_long['Code_Long'].fillna('CLASS', inplace = True)
    df_beta_long.set_index(['DatePoint', 'Class', 'Code_Long', 'Code_Short'], inplace = True)
    df_beta_long.sort_index(axis = 0, level = ['DatePoint', 'Class', 'Code_Long'], inplace = True)
    ### Exporting geo data:
    import geopandas as gpd
    file_countries_gdf = gpd.read_file(path_countries_map_shp)[['ADMIN', 'ADM0_A3', 'geometry']]
    file_countries_gdf.columns = ['Country_Name', 'Country_Long_Code', 'Country_Geometry']
    file_countries_gdf.sort_values('Country_Long_Code', inplace = True)
    file_countries_gdf.drop(file_countries_gdf[file_countries_gdf['Country_Name'] == 'Antarctica'].index, axis = 0, inplace = True)
    ### Converting polygons to coordinates arrays dataframe for bokech patches:
    from shapely.geometry import MultiPolygon, Polygon
    arr_names = []
    arr_codes = []
    arr_arr_x = []
    arr_arr_y = []
    ### Extacting polygons from gdf file:
    for country_counter, (country_name, country_long_code, country_geometry) in file_countries_gdf.iterrows():
        ### For group of polygons:
        if isinstance(country_geometry,  MultiPolygon):
            for country_polygon in country_geometry:
                arr_names.append(country_name)            
                arr_codes.append(country_long_code)
                arr_arr_x.append(list(country_polygon.exterior.coords.xy[0]))
                arr_arr_y.append(list(country_polygon.exterior.coords.xy[1]))             
        ### For single polygons:            
        else:
            country_polygon = country_geometry
            arr_names.append(country_name)               
            arr_codes.append(country_long_code)
            arr_arr_x.append(list(country_polygon.exterior.coords.xy[0]))
            arr_arr_y.append(list(country_polygon.exterior.coords.xy[1]))  
    ### Constructing dataframe with row for each polygon:            
    tup_country_coords = tuple(zip(arr_names, arr_codes, arr_arr_x, arr_arr_y))
    df_country_coords = pd.DataFrame(list(tup_country_coords), columns = ['Country_Name', 'Country_Long_Code', 'Coord_X_Array', 'Coord_Y_Array'])
    df_country_coords.set_index(['Country_Long_Code', 'Country_Name'], inplace = True)
    ### Creating data table for class evolution through month/year to further connect with Slider bokeh widget:
    df_class_evolution = df_beta_long.reset_index()
    df_class_evolution = df_class_evolution[['DatePoint', 'Class', 'Code_Long', 'None', 'By_Date', 'By_Value']]
    df_class_evolution['Year_Month'] = df_class_evolution['DatePoint'].dt.to_period('M')
    df_class_evolution.drop('DatePoint', axis = 1, inplace = True)
    df_class_evolution.set_index(['Year_Month'], inplace = True)
    ### Preparing dictionary of class membership for world map countries on each available month/year:
    arr_classes_name = np.append(df_class_evolution['Class'].unique(), 'ALL')
    arr_weights_name = np.array(['None', 'By_Date', 'By_Value'])
    dict_date_classes = {}
    dict_beta_general = {}
    for iter_class_name in arr_classes_name:
        dict_beta_general[iter_class_name] = {}
        for iter_weight_name in arr_weights_name:
            dict_beta_general[iter_class_name][iter_weight_name] = {}    
    dict_iter_country = {}
    for iter_index in df_class_evolution.index.unique():
        df_iter_class_plus = df_class_evolution.loc[iter_index]
        df_iter_class_plus = df_iter_class_plus.reset_index()
        df_iter_class_plus.set_index('Code_Long', inplace = True)
        df_iter_class_plus.drop('Year_Month', axis = 1 ,inplace = True)
        df_iter_country = df_country_coords.reset_index(level = [1])['Country_Name'].to_frame().merge(df_iter_class_plus, how = 'left', 
                                                                                                      left_on = 'Country_Long_Code', right_index = True)
        df_iter_country.drop('Country_Name', axis = 1, inplace = True)
        dict_date_classes[iter_index.strftime('%Y-%m')] = df_iter_country['Class'].values  
        for iter_class_name in arr_classes_name:
            df_iter_country_class = df_iter_country.copy()
            if (iter_class_name != 'ALL'):
                df_iter_country_class.loc[df_iter_country[df_iter_country['Class'] != iter_class_name].index, ['None', 'By_Date', 'By_Value']] = np.NaN           
            dict_iter_country[iter_class_name] = df_iter_country_class
            for iter_weight_name in arr_weights_name:
                dict_beta_general[iter_class_name][iter_weight_name][iter_index.strftime('%Y-%m')] = dict_iter_country[iter_class_name][iter_weight_name].values
    ### Configuring latest known MSCI data as initial world map classification data source:
    df_current_date = df_class_evolution.loc[df_class_evolution.index[-1]]
    df_current_date = df_current_date.reset_index()
    df_current_date.set_index('Code_Long', inplace = True)
    df_current_date.drop('Year_Month', axis = 1 ,inplace = True)
    df_country_coords_class = df_country_coords.merge(df_current_date, how = 'left', left_on = 'Country_Long_Code', right_index = True)
    df_country_coords_class['Active'] = df_country_coords_class['None']
    ### Configuring actual MSCI common average as initial class average beta plot data source:
    df_class_average = df_class_evolution[df_class_evolution['Class'] == 'ALL'].copy()
    df_class_average['Active'] = df_class_average['None']
    ### Creating list for countrty selection:
    df_beta_countries = df_class_evolution.reset_index().merge(df_country_codes.reset_index(), how = 'left', left_on = 'Code_Long', right_on = 'ISO LONG')
    df_countries_list = df_beta_countries.drop_duplicates(subset = ['COUNTRY']).dropna().reset_index()
    df_countries_list.sort_values('COUNTRY', inplace = True)
    ### Reconfiguring membership history for long ISO codes
    df_history_membership_long = df_history_membership.merge(df_country_codes, how = 'left', left_index = True, right_on = 'ISO SHORT').reset_index()
    df_history_membership_long.head()
    df_history_membership_long.drop(['COUNTRY', 'ISO SHORT'], axis = 1, inplace = True)
    df_history_membership_long.rename(columns = {'ISO LONG': 'Member Code'}, inplace = True)
    df_history_membership_long.set_index('Member Code')
    ### Creating transit dataframe as container for data transfer betwwen widgets and figures:
    df_meta_transit = pd.DataFrame([str(df_class_evolution.index[-1]), 'ALL', 'None', 'None'], index = ['Year_Month', 'Class', 'Weights', 'Country'])  
    
    ### Defining output to notebook or to html file:
    b_pl.output_notebook()
    ### Bokeh Data Source defining:
    src_meta_transit =  b_pl.ColumnDataSource(df_meta_transit.T) ### For data transfer betwwen elements
    src_country_coords =  b_pl.ColumnDataSource(df_country_coords_class.reset_index()) ### For world map borders
    src_beta_evolution = b_pl.ColumnDataSource(df_class_evolution.reset_index()) ### For plotting country level beta values (common data)
    src_class_average = b_pl.ColumnDataSource(df_class_average.reset_index()) ### For plotting class average beta values
    src_country_chosen = b_pl.ColumnDataSource(df_class_average.copy().reset_index()) ### For plotting country level beta values (chosen country)
    src_class_boxes = b_md.ColumnDataSource(df_history_membership_long) ### For background color box annotations
    ### Plot boundary dates defining:
    fig_min_date = df_class_average.reset_index()['Year_Month'].min().to_timestamp()
    fig_max_date = df_class_average.reset_index()['Year_Month'].max().to_timestamp()
    ### Weights select additional data connectors constructing:
    dict_weights_select = {'Priority: none (equal weights)': 'None', 'Priority: by date': 'By_Date', 'Priority: by similarity to current value (MRI vector)': 'By_Value'}
    dict_weights_mirror = dict(zip(dict_weights_select.values(), dict_weights_select.keys()))
    arr_weights_select = list(dict_weights_select.keys())
    ### Class select additional data connectors constructing:
    dict_classes_select = {'Developed Markets': 'DM', 'Emerging Markets': 'EM', 'Frontier Markets': 'FM', 'Standalone Markets': 'SM', 'All Markets': 'ALL'}
    dict_classes_mirror = dict(zip(dict_classes_select.values(), dict_classes_select.keys()))
    arr_classes_select = list(dict_classes_select.keys())
    ### Country select additional data connectors constructing:    
    arr_country_select_list = list(np.append((df_countries_list['COUNTRY'] + ' (' + df_countries_list['Code_Long'] + ')').values, 'No country selected'))
    arr_country_code_list = list(np.append(df_countries_list['Code_Long'].values, 'None'))
    dict_country_list = dict(zip(arr_country_select_list, arr_country_code_list))
    dict_country_mirror = dict(zip(arr_country_code_list, arr_country_select_list))
    ### Colors categorising and mapping:
    linear_cm_beta = b_md.LinearColorMapper(low = -0.35, high = 0.10, palette = inferno(45), nan_color = 'silver')
    arr_classes_name = list(dict_classes_select.values())[: -1][::-1]
    arr_classes_num = list(list(np.arange(len(arr_classes_name)) + 0.5))
    arr_classes_str = list(map(str, arr_classes_num))
    dict_classes_label = dict(zip(arr_classes_str, arr_classes_name))
    dict_classes_colors = dict(zip(arr_classes_name, Set2[4]))
    categorical_cm_class = b_md.CategoricalColorMapper(factors = arr_classes_name, palette = Set2[4], nan_color = 'silver')
    linear_cm_class = b_md.LinearColorMapper(low = 0, high = len(arr_classes_name), palette = Set2[4], nan_color = 'silver')
    ### Initialising beta values plot figure:
    str_fig_beta_toolbar =  'pan, wheel_zoom, reset'
    tup_fig_beta_size = (800, 200)
    str_fig_beta_title = 'Regressor (MRI) beta: '
    fig_beta_plot = b_pl.figure(tools = str_fig_beta_toolbar, active_scroll = 'wheel_zoom', plot_width = tup_fig_beta_size[0], plot_height = tup_fig_beta_size[1],
                                  title = str_fig_beta_title + dict_classes_mirror['ALL'] + ' / ' + dict_weights_mirror['None'],
                                  x_axis_type = 'datetime', x_range = (fig_min_date, fig_max_date)) 
    ### Drawing beta plots:
    line_class_plot = fig_beta_plot.line(x = 'Year_Month', y = 'Active', source = src_class_average,
                                           line_color = 'blue', line_width = 2.0, line_dash = 'dotted')
    line_country_plot = fig_beta_plot.line(x = 'Year_Month', y = 'Active', source = src_country_chosen,
                                             line_color = 'red', line_width = 2.0, line_dash = 'dotted', line_alpha = 0.0)
    ### Adding classes color bar to beta plot figure:
    color_bar_class = b_md.ColorBar(color_mapper = linear_cm_class, scale_alpha = 0.2, label_standoff = 8, width = 20, height = tup_fig_beta_size[1] - 50,
                                    border_line_color = None, orientation = 'vertical', location = (0, -10),
                                    ticker = b_md.FixedTicker(ticks = arr_classes_num), major_label_overrides = dict_classes_label,
                                    title = 'Class')
    fig_beta_plot.add_layout(color_bar_class, 'right')
    ### Adding current data marker to beta plot figure:    
    span_current_date = b_md.Span(location = fig_max_date, dimension = 'height', line_color = 'gray', line_dash = 'dashed', line_width = 3.0)
    fig_beta_plot.add_layout(span_current_date)
    ### Adding dynamic legend to beta plot figure:     
    legend_beta_plot = b_md.Legend(items = [('Class Average', [line_class_plot]), (arr_country_select_list[-1] , [line_country_plot])], location = (0, 0))
    fig_beta_plot.add_layout(legend_beta_plot)
    ### Adding array of blank annotations (class evolution stages) to beta plot figure:
    num_annotations = 5
    arr_annotations = []
    for counter_annotation in np.arange(num_annotations):
        arr_annotations.append(b_md.BoxAnnotation(left = fig_max_date, right = fig_max_date, fill_alpha = 0.2, fill_color = 'white'))
        fig_beta_plot.add_layout(arr_annotations[counter_annotation])
    ### Tuning beta plot figure:
    fig_beta_plot.legend.location = 'top_left'
    fig_beta_plot.legend.background_fill_alpha  = 0.75
    fig_beta_plot.yaxis.formatter = b_md.NumeralTickFormatter(format = '.00')
    ### Creating hover inspection to beta plot figure:
    fig_beta_plot.add_tools(b_md.HoverTool(tooltips = [('Country code', '@Code_Long'), ('Month', '@Year_Month{%Y-%m}'), ('Regressor beta (equal weights)', '@None{0,0.000}'),
                                                         ('Regressor beta (by date weights)', '@By_Date{0,0.000}'), ('Regressor beta (similarity weights)', '@By_Value{0,0.000}')], 
                                             formatters = {'None': 'numeral', 'By_Date': 'numeral', 'By_Value': 'numeral', 'Year_Month': 'datetime'}))
    ### Constructing select widget for choosing country tuning (including connection between selected value and plot):
    callback_country_select = CustomJS(args = dict(fig_plot_to_update = fig_beta_plot, legend_beta = legend_beta_plot, line_country = line_country_plot,
                                                   arr_country_codes = dict_country_list,
                                                   cds_transit = src_meta_transit, cds_country_plot = src_country_chosen, cds_full = src_beta_evolution,
                                                   cds_annotations = src_class_boxes, date_min = fig_min_date, date_max = fig_max_date,
                                                   arr_annotations = arr_annotations, num_annotations = num_annotations, dict_colors = dict_classes_colors),
                                       code = """                                  
                                              var country_chosen = cb_obj.value;
                                              var country_code = arr_country_codes[country_chosen];
                                              cds_transit.data['Country'] = country_code;
                                              var weights_selected = cds_transit.data['Weights'];                                
                                              if (country_code == 'None')
                                              {
                                                  line_country.glyph.line_alpha = 0.0;                                             
                                              }
                                              else
                                              {
                                                  line_country.glyph.line_alpha = 1.0;                                           
                                              }
                                              var legend_beta_last = (legend_beta.items.length - 1);
                                              legend_beta.items[legend_beta_last].label['value'] = country_chosen;
                                              legend_beta.change.emit();  
                                              cds_country_plot.data['Year_Month'] = [];
                                              cds_country_plot.data['Class'] = [];
                                              cds_country_plot.data['Code_Long'] = [];
                                              cds_country_plot.data['None'] = [];
                                              cds_country_plot.data['By_Date'] = [];
                                              cds_country_plot.data['By_Value'] = [];
                                              cds_country_plot.data['Active'] = []; 
                                              for (var iter_counter = 0; iter_counter <= cds_full.get_length(); iter_counter++)
                                              {
                                                  if ((cds_full.data['Code_Long'][iter_counter] == country_code))
                                                  {
                                                      cds_country_plot.data['Year_Month'].push(cds_full.data['Year_Month'][iter_counter]);
                                                      cds_country_plot.data['Class'].push(cds_full.data['Class'][iter_counter]);
                                                      cds_country_plot.data['Code_Long'].push(cds_full.data['Code_Long'][iter_counter]);
                                                      cds_country_plot.data['None'].push(cds_full.data['None'][iter_counter]);
                                                      cds_country_plot.data['By_Date'].push(cds_full.data['By_Date'][iter_counter]);
                                                      cds_country_plot.data['By_Value'].push(cds_full.data['By_Value'][iter_counter]);
                                                      cds_country_plot.data['Active'].push(cds_full.data[weights_selected][iter_counter]);
                                                  }
                                              }                                          
                                              cds_country_plot.change.emit();
                                              var iter_ann;
                                              for (iter_ann = 0; iter_ann < num_annotations; iter_ann++)
                                              {
                                                arr_annotations[iter_ann].fill_color = 'white';
                                                arr_annotations[iter_ann].left = date_max;
                                                arr_annotations[iter_ann].right = date_max;
                                              }
                                              iter_ann = 0;
                                              var iter_history;
                                              for (iter_history = 0; iter_history <= cds_annotations.get_length(); iter_history++)
                                              {
                                                if (cds_annotations.data['Member Code'][iter_history] == country_code)
                                                {
                                                    arr_annotations[iter_ann].fill_color = dict_colors[cds_annotations.data['Index Name'][iter_history]];
                                                    arr_annotations[iter_ann].left = cds_annotations.data['Start Date'][iter_history];
                                                    arr_annotations[iter_ann].right = cds_annotations.data['End Date'][iter_history];
                                                    iter_ann = iter_ann + 1;
                                                }
                                              }                                                  
                                              fig_plot_to_update.change.emit(); 
                                              """)
    select_country = Select(title = 'Select country to show plot:', options = arr_country_select_list, value = arr_country_select_list[-1], callback = callback_country_select)
    ### Initialising worldmap classes evolution figure:
    str_fig_worldmap_toolbar =  'pan, wheel_zoom, reset'
    tup_fig_worldmap_size = (tup_fig_beta_size[0], tup_fig_beta_size[1] * 2)
    str_fig_worldmap_title = 'MSCI Membership regressor (MRI) beta graduation'
    fig_world_map = b_pl.figure(tools = str_fig_worldmap_toolbar, active_scroll = 'wheel_zoom', plot_width = tup_fig_worldmap_size[0], plot_height = tup_fig_worldmap_size[1],
                                title = str_fig_worldmap_title)
    ### Tuning worldmap figure:
    fig_world_map.axis.visible = False
    fig_world_map.xgrid.visible = False
    fig_world_map.ygrid.visible = False
    fig_world_map.toolbar.autohide = True
    ### Creating hover inspection to worldmap figure:    
    fig_world_map.add_tools(b_md.HoverTool(tooltips = [('Country', '@Country_Name'), ('MSCI Class', '@Class'), ('Regressor beta (equal weights)', '@None{0,0.000}'),
                                                       ('Regressor beta (by date weights)', '@By_Date{0,0.000}'), ('Regressor beta (similarity weights)', '@By_Value{0,0.000}')], 
                                           formatters = {'None': 'numeral', 'By_Date': 'numeral', 'By_Value': 'numeral'}))
    ### Drawing world map:
    patches_world_map = fig_world_map.patches('Coord_X_Array', 'Coord_Y_Array', source = src_country_coords,
                                              color = {'field': 'Active', 'transform': linear_cm_beta}, fill_alpha = 1.0, line_color = 'lightgray', 
                                              line_width = 1.0)
    ### Adding beta value color bar to beta plot figure: 
    color_bar_class = b_md.ColorBar(color_mapper = linear_cm_beta, label_standoff = 8, width = 20, height = tup_fig_worldmap_size[1] - 100,
                                    border_line_color = None, orientation = 'vertical', location = (0, 50),
                                    ticker = b_md.AdaptiveTicker(desired_num_ticks = int(len(linear_cm_beta.palette) / 5)),
                                    formatter = b_md.NumeralTickFormatter(format = '.00'))
    fig_world_map.add_layout(color_bar_class, 'right')
    ### Drawing and tuning date slider widget, creating a trigger for it's changes:
    callback_date_slider = CustomJS(args = dict(fig_map_to_update = fig_world_map, title_map_main_part = str_fig_worldmap_title, 
                                                fig_plot_to_update = fig_beta_plot, span_plot = span_current_date,
                                                cds_world = src_country_coords, cds_transit = src_meta_transit, cds_plot = src_beta_evolution,
                                                arr_date_classes = dict_date_classes, arr_date_beta_general = dict_beta_general),
                                    code = """                             
                                           var date_chosen = new Date(cb_obj.value);
                                           var class_selected = cds_transit.data['Class'];
                                           var weights_selected = cds_transit.data['Weights'];
                                           var arr_months = ['January', 'February', 'March', 'April', 'May', 'June', 
                                                             'July', 'August', 'September', 'October', 'November', 'December'];
                                           var date_month_name = arr_months[date_chosen.getMonth()]; 
                                           fig_map_to_update.title.text = title_map_main_part + ': ' + date_chosen.getFullYear() + ' / ' + date_month_name;
                                           var date_year_month = date_chosen.getFullYear().toString();
                                           if (date_chosen.getMonth() > 8)
                                           {
                                               date_year_month = date_year_month + '-' + (date_chosen.getMonth() + 1).toString();
                                           }
                                           else
                                           {
                                               date_year_month = date_year_month + '-' + '0' + (date_chosen.getMonth() + 1).toString();
                                           }                                  
                                           cds_transit.data['Year_Month'] = date_year_month;                                       
                                           cds_world.data['Class'] = arr_date_classes[date_year_month];
                                           cds_world.data['None'] = arr_date_beta_general[class_selected]['None'][date_year_month];
                                           cds_world.data['By_Date'] = arr_date_beta_general[class_selected]['By_Date'][date_year_month];
                                           cds_world.data['By_Value'] = arr_date_beta_general[class_selected]['By_Value'][date_year_month];                         
                                           cds_world.data['Active'] = cds_world.data[weights_selected];
                                           cds_world.change.emit();                               
                                           cds_transit.change.emit(); 
                                           fig_map_to_update.change.emit();           
                                           span_plot.location = cb_obj.value;
                                           span_plot.change.emit();
                                           fig_plot_to_update.change.emit();                                       
                                           """)
    date_min = df_beta_long.index.get_level_values(level = 0).min()
    date_max = df_beta_long.index.get_level_values(level = 0).max()
    slider_dates = DateSlider(title = 'Date to show classes membership', width = tup_fig_worldmap_size[0] - 50, start = date_min, end = date_max, step = 1, value = date_max)
    slider_dates.callback_policy = 'throttle'
    slider_dates.js_on_change('value', callback_date_slider)
    slider_dates.tooltips = False
    ### Creating select widget for choosing class tuning (including connection between selected value and range slider boundaries, figure plots, figure boundaries, figure title):
    callback_class_select = CustomJS(args = dict(fig_map_to_update = fig_world_map, fig_plot_to_update = fig_beta_plot, title_plot_main_part = str_fig_beta_title,
                                                 cds_world = src_country_coords, cds_transit = src_meta_transit, 
                                                 cds_full = src_beta_evolution, cds_class_plot = src_class_average,
                                                 arr_classes_names = dict_classes_select, arr_date_beta_general = dict_beta_general,
                                                 arr_classes_mirror = dict_classes_mirror, arr_weights_mirror = dict_weights_mirror),
                                     code = """
                                            var date_year_month = cds_transit.data['Year_Month'];
                                            var class_selected = arr_classes_names[cb_obj.value];
                                            var weights_selected = cds_transit.data['Weights'];                                        
                                            cds_world.data['None'] = arr_date_beta_general[class_selected]['None'][date_year_month];
                                            cds_world.data['By_Date'] = arr_date_beta_general[class_selected]['By_Date'][date_year_month];
                                            cds_world.data['By_Value'] = arr_date_beta_general[class_selected]['By_Value'][date_year_month];                         
                                            cds_world.data['Active'] = cds_world.data[weights_selected];     
                                            cds_transit.data['Class'] = class_selected;                                           
                                            cds_world.change.emit();                               
                                            cds_transit.change.emit(); 
                                            fig_map_to_update.change.emit();
                                            fig_plot_to_update.title.text = title_plot_main_part + arr_classes_mirror[class_selected] + ' / '
                                            fig_plot_to_update.title.text = fig_plot_to_update.title.text + arr_weights_mirror[weights_selected];
                                            cds_class_plot.data['Year_Month'] = [];
                                            cds_class_plot.data['Class'] = [];
                                            cds_class_plot.data['Code_Long'] = [];
                                            cds_class_plot.data['None'] = [];
                                            cds_class_plot.data['By_Date'] = [];
                                            cds_class_plot.data['By_Value'] = [];
                                            cds_class_plot.data['Active'] = [];                                        
                                            for (var iter_counter = 0; iter_counter <= cds_full.get_length(); iter_counter++)
                                            {
                                                if ((cds_full.data['Class'][iter_counter] == class_selected) && (cds_full.data['Code_Long'][iter_counter] == 'CLASS'))
                                                {
                                                    cds_class_plot.data['Year_Month'].push(cds_full.data['Year_Month'][iter_counter]);
                                                    cds_class_plot.data['Class'].push(cds_full.data['Class'][iter_counter]);
                                                    cds_class_plot.data['Code_Long'].push(cds_full.data['Code_Long'][iter_counter]);
                                                    cds_class_plot.data['None'].push(cds_full.data['None'][iter_counter]);
                                                    cds_class_plot.data['By_Date'].push(cds_full.data['By_Date'][iter_counter]);
                                                    cds_class_plot.data['By_Value'].push(cds_full.data['By_Value'][iter_counter]);
                                                    cds_class_plot.data['Active'].push(cds_full.data[weights_selected][iter_counter]);
                                                }
                                            }
                                            cds_class_plot.change.emit();                                     
                                            fig_plot_to_update.change.emit();
                                            """)
    select_class = Select(title = 'Select MSCI class to show:', options = arr_classes_select, value = arr_classes_select[-1], callback = callback_class_select)
    ### Creating select widget for choosing country (including connection between selected value and range slider boundaries, figure plots, figure boundaries, figure title):
    callback_weights_select = CustomJS(args = dict(fig_map_to_update = fig_world_map, fig_plot_to_update = fig_beta_plot, title_plot_main_part = str_fig_beta_title, 
                                                   cds_world = src_country_coords, cds_transit = src_meta_transit, 
                                                   cds_full = src_beta_evolution, cds_class_plot = src_class_average, cds_country_plot = src_country_chosen,
                                                   arr_weights_names = dict_weights_select, arr_date_beta_general = dict_beta_general,                                      
                                                   arr_classes_mirror = dict_classes_mirror, arr_weights_mirror = dict_weights_mirror),
                                       code = """
                                              var date_year_month = cds_transit.data['Year_Month'];
                                              var class_selected = cds_transit.data['Class'];
                                              var country_code = cds_transit.data['Country'];                                                                                  
                                              var weights_selected = arr_weights_names[cb_obj.value];                                            
                                              cds_world.data['Active'] = cds_world.data[weights_selected];
                                              cds_transit.data['Weights'] = weights_selected;
                                              cds_transit.change.emit();
                                              cds_world.change.emit();                                       
                                              fig_map_to_update.change.emit();
                                              fig_plot_to_update.title.text = title_plot_main_part + arr_classes_mirror[class_selected] + ' / '
                                              fig_plot_to_update.title.text = fig_plot_to_update.title.text + arr_weights_mirror[weights_selected];
                                              cds_class_plot.data['Year_Month'] = [];
                                              cds_class_plot.data['Class'] = [];
                                              cds_class_plot.data['Code_Long'] = [];
                                              cds_class_plot.data['None'] = [];
                                              cds_class_plot.data['By_Date'] = [];
                                              cds_class_plot.data['By_Value'] = [];
                                              cds_class_plot.data['Active'] = [];                                        
                                              for (var iter_counter = 0; iter_counter <= cds_full.get_length(); iter_counter++)
                                              {
                                                  if ((cds_full.data['Class'][iter_counter] == class_selected) && (cds_full.data['Code_Long'][iter_counter] == 'CLASS'))
                                                  {
                                                      cds_class_plot.data['Year_Month'].push(cds_full.data['Year_Month'][iter_counter]);
                                                      cds_class_plot.data['Class'].push(cds_full.data['Class'][iter_counter]);
                                                      cds_class_plot.data['Code_Long'].push(cds_full.data['Code_Long'][iter_counter]);
                                                      cds_class_plot.data['None'].push(cds_full.data['None'][iter_counter]);
                                                      cds_class_plot.data['By_Date'].push(cds_full.data['By_Date'][iter_counter]);
                                                      cds_class_plot.data['By_Value'].push(cds_full.data['By_Value'][iter_counter]);
                                                      cds_class_plot.data['Active'].push(cds_full.data[weights_selected][iter_counter]);
                                                  }
                                              }
                                              cds_class_plot.change.emit();
                                              cds_country_plot.data['Year_Month'] = [];
                                              cds_country_plot.data['Class'] = [];
                                              cds_country_plot.data['Code_Long'] = [];
                                              cds_country_plot.data['None'] = [];
                                              cds_country_plot.data['By_Date'] = [];
                                              cds_country_plot.data['By_Value'] = [];
                                              cds_country_plot.data['Active'] = []; 
                                              for (var iter_counter = 0; iter_counter <= cds_full.get_length(); iter_counter++)
                                              {
                                                  if ((cds_full.data['Code_Long'][iter_counter] == country_code))
                                                  {
                                                      cds_country_plot.data['Year_Month'].push(cds_full.data['Year_Month'][iter_counter]);
                                                      cds_country_plot.data['Class'].push(cds_full.data['Class'][iter_counter]);
                                                      cds_country_plot.data['Code_Long'].push(cds_full.data['Code_Long'][iter_counter]);
                                                      cds_country_plot.data['None'].push(cds_full.data['None'][iter_counter]);
                                                      cds_country_plot.data['By_Date'].push(cds_full.data['By_Date'][iter_counter]);
                                                      cds_country_plot.data['By_Value'].push(cds_full.data['By_Value'][iter_counter]);
                                                      cds_country_plot.data['Active'].push(cds_full.data[weights_selected][iter_counter]);
                                                  }
                                              }                                          
                                              cds_country_plot.change.emit();  
                                              fig_plot_to_update.change.emit();                                          
                                              """)
    select_weights = Select(title = 'Select way of weights allocation:', options = arr_weights_select, value = arr_weights_select[0], callback = callback_weights_select)
    ### Constructing common layout: 
    layout_world = b_col(b_row(widgetbox(select_class), widgetbox(select_weights)), fig_world_map, widgetbox(slider_dates), fig_beta_plot, widgetbox(select_country))

    return layout_world


def hh_msci_factors(MSCI_returns_path, MSCI_returns_key, arr_factors, df_beta_all, df_expvol_all, period_shift = 1):
    """
    Version 0.02 2019-07-09
    
    FUNCTIONALITY: 
      Creating factors data table for MSCI returns
    OUTPUT:
      ser_returns (pd.Series) - source returns data vector
      dict_factor_pairs_container (array of pd.Dataframe) - array of factor data tables
    INPUT:
      MSCI_returns_path (string) - path to the MSCI returns HDF5 file
      MSCI_returns_key (string) - data object key to access MSCI USD monthly returns from HDF5 file 
      arr_factors (array) - list of string factors for comparision with returns(t + period_shift), such as:
          'month_back' - returns(t);
          'year_back' - ((1 + returns(t - 11)) * ... * (1 + returns(t)) - 1);
          'echo_back' - ((1 + returns(t - 8)) * ... * (1 + returns(t - 3)) - 1);
          'mri_beta_equal_weighted' - MRI equal weighted regression beta;
          'mri_beta_date_weighted' - MRI date weighted regression beta;
          'mri_beta_cond_weighted' - MRI similarity weighted regression beta;          
          'expvol_beta_equal_weighted' - equal weighted exponential volatility;
          'expvol_beta_date_weighted' - date weighted exponential volatility;
          'expvol_beta_cond_weighted' - MRI similarity exponential volatility;          
          'volatility_surprise' - -ln([Date weighted exponential volatility] / [MRI similarity exponential volatility]);      
      df_beta_all (pd.DataFrame) - table of MSCI on MRI regression betas
      df_expvol_all (pd.DataFrame) - table of MSCI exponential volatilites
      period_shift (number) - quantity of months to step back for efficacy measures calculating        
    """    
    
    import numpy as np
    import pandas as pd    
    
    ### Preparing MSCI data for looping:
    df_returns = pd.read_hdf(MSCI_returns_path, MSCI_returns_key)    
    df_returns.reset_index(level = 'Country', drop = True, inplace = True)
    df_returns.drop('INDEX', level = 'Code', inplace = True)
    df_returns = df_returns.swaplevel()
    df_returns.sort_index(level = [0, 1], inplace = True)
    ser_returns = df_returns.squeeze()   
    ### From based 100 returns to month-to-month returns
    for country_iter in ser_returns.index.get_level_values(level = 0).unique():
        ser_returns[country_iter] = (ser_returns[country_iter] / ser_returns[country_iter].shift(1) - 1)
    ser_returns.fillna(0, inplace = True)
    ser_returns = ser_returns.swaplevel(copy = False)
    ser_returns.sort_index(inplace = True)
    ser_returns = ser_returns[ser_returns != 0]
    ### MRI betas and MSCI expvol preparation:
    df_beta_all = df_beta_all.swaplevel()
    df_beta_all.sort_index(inplace = True)
    df_expvol_all = df_expvol_all.swaplevel()
    df_expvol_all.sort_index(inplace = True)
    ### Looping containers:
    dict_factor_pairs_container = {}
    dict_factor_measures_container = {}
    ### Kreating constants
    num_year_months = 12
    ### Creating "Next Returns" containing table to be merged with factor:
    df_month_back = ser_returns.to_frame().reset_index(level = 1)
    df_month_back.index = df_month_back.index.shift(-period_shift, 'BM')    
    ### Looping factors:
    for iter_factor in arr_factors:
        ### Generating table for "Now / Next" factor:
        if (iter_factor == 'month_back'):
            df_iter_base = ser_returns.to_frame().reset_index()
            df_iter_base = df_iter_base.merge(df_month_back.reset_index(), how = 'left', left_on = ['Date', 'Code'], right_on = ['Date', 'Code'])
            df_iter_base.rename(columns = {'Returns_x': 'Factor', 'Returns_y': 'Next Returns'}, inplace = True)      
            df_iter_base = df_iter_base.set_index(['Date', 'Code'])   
        ### Generating table for "Year Back from Now / Next" factor:
        if (iter_factor == 'year_back'):
            df_iter_base = ser_returns.to_frame()
            df_iter_base['Factor'] = np.NaN
            for iter_index in ser_returns.index:
                ser_year_back = ser_returns.loc[iter_index[0] - pd.offsets.BMonthEnd(num_year_months - 1) : iter_index[0], iter_index[1]]
                if (len(ser_year_back) == num_year_months):
                    df_iter_base.loc[iter_index[0], iter_index[1]]['Factor'] = (ser_year_back + 1).prod() - 1
            df_iter_base = df_iter_base.reset_index()
            df_iter_base = df_iter_base.merge(df_month_back.reset_index(), how = 'left', left_on = ['Date', 'Code'], right_on = ['Date', 'Code'])            
            df_iter_base.drop('Returns_x', axis = 1, inplace = True)
            df_iter_base.rename(columns = {'Returns_y': 'Next Returns'}, inplace = True)            
            df_iter_base = df_iter_base.set_index(['Date', 'Code'])
        ### Generating table for "Echo Back from Now / Next" factor:
        if (iter_factor == 'echo_back'):
            df_iter_base = ser_returns.to_frame()
            df_iter_base['Factor'] = np.NaN
            for iter_index in ser_returns.index:
                start_index = iter_index[0] - pd.offsets.BMonthEnd(num_year_months * 3 / 4 - 1)
                end_index = iter_index[0] - pd.offsets.BMonthEnd(num_year_months * 1 / 4)
                ser_echo_back = ser_returns.loc[start_index : end_index, iter_index[1]]
                if (len(ser_echo_back) == round(num_year_months / 2)):
                    df_iter_base.loc[iter_index[0], iter_index[1]]['Factor'] = (ser_echo_back + 1).prod() - 1
            df_iter_base = df_iter_base.reset_index()
            df_iter_base = df_iter_base.merge(df_month_back.reset_index(), how = 'left', left_on = ['Date', 'Code'], right_on = ['Date', 'Code'])            
            df_iter_base.drop('Returns_x', axis = 1, inplace = True)
            df_iter_base.rename(columns = {'Returns_y': 'Next Returns'}, inplace = True)            
            df_iter_base = df_iter_base.set_index(['Date', 'Code'])
        ### Generating table for "MRI regression Beta Equal Weighted for Now / Next" factor:
        if (iter_factor == 'mri_beta_equal_weighted'):
            df_iter_base = pd.DataFrame(np.NaN, index = ser_returns.index, columns = [])
            df_iter_base = df_iter_base.reset_index()
            df_iter_base = df_iter_base.merge(df_beta_all.reset_index(), how = 'left', left_on = ['Date', 'Code'], right_on = ['DatePoint', 'Country'])
            df_iter_base.drop(axis = 1, labels = ['Country', 'DatePoint', 'By_Date', 'By_Value'], inplace = True)
            df_iter_base = df_iter_base.merge(df_month_back.reset_index(), how = 'left', left_on = ['Date', 'Code'], right_on = ['Date', 'Code'])       
            df_iter_base.rename(columns = {'None': 'Factor', 'Returns': 'Next Returns'}, inplace = True)            
            df_iter_base = df_iter_base.set_index(['Date', 'Code'])            
        ### Generating table for "MRI regression Beta Date Weighted for Now / Next" factor:
        if (iter_factor == 'mri_beta_date_weighted'):
            df_iter_base = pd.DataFrame(np.NaN, index = ser_returns.index, columns = [])
            df_iter_base = df_iter_base.reset_index()
            df_iter_base = df_iter_base.merge(df_beta_all.reset_index(), how = 'left', left_on = ['Date', 'Code'], right_on = ['DatePoint', 'Country'])
            df_iter_base.drop(axis = 1, labels = ['Country', 'DatePoint', 'None', 'By_Value'], inplace = True)
            df_iter_base = df_iter_base.merge(df_month_back.reset_index(), how = 'left', left_on = ['Date', 'Code'], right_on = ['Date', 'Code'])       
            df_iter_base.rename(columns = {'By_Date': 'Factor', 'Returns': 'Next Returns'}, inplace = True)            
            df_iter_base = df_iter_base.set_index(['Date', 'Code'])     
        ### Generating table for "MRI regression Beta Conditional Weighted for Now / Next" factor:
        if (iter_factor == 'mri_beta_cond_weighted'):
            df_iter_base = pd.DataFrame(np.NaN, index = ser_returns.index, columns = [])
            df_iter_base = df_iter_base.reset_index()
            df_iter_base = df_iter_base.merge(df_beta_all.reset_index(), how = 'left', left_on = ['Date', 'Code'], right_on = ['DatePoint', 'Country'])
            df_iter_base.drop(axis = 1, labels = ['Country', 'DatePoint', 'None', 'By_Date'], inplace = True)
            df_iter_base = df_iter_base.merge(df_month_back.reset_index(), how = 'left', left_on = ['Date', 'Code'], right_on = ['Date', 'Code'])       
            df_iter_base.rename(columns = {'By_Value': 'Factor', 'Returns': 'Next Returns'}, inplace = True)            
            df_iter_base = df_iter_base.set_index(['Date', 'Code'])         
        ### Generating table for "Exponential Volatility Equal Weighted for Now / Next" factor:
        if (iter_factor == 'expvol_equal_weighted'):
            df_iter_base = pd.DataFrame(np.NaN, index = ser_returns.index, columns = [])
            df_iter_base = df_iter_base.reset_index()
            df_iter_base = df_iter_base.merge(df_expvol_all.reset_index(), how = 'left', left_on = ['Date', 'Code'], right_on = ['DatePoint', 'Country'])
            df_iter_base.drop(axis = 1, labels = ['Country', 'DatePoint', 'By_Date', 'By_Value'], inplace = True)
            df_iter_base = df_iter_base.merge(df_month_back.reset_index(), how = 'left', left_on = ['Date', 'Code'], right_on = ['Date', 'Code'])       
            df_iter_base.rename(columns = {'None': 'Factor', 'Returns': 'Next Returns'}, inplace = True)            
            df_iter_base = df_iter_base.set_index(['Date', 'Code'])            
        ### Generating table for "Exponential Volatility Date Weighted for Now / Next" factor:
        if (iter_factor == 'expvol_date_weighted'):
            df_iter_base = pd.DataFrame(np.NaN, index = ser_returns.index, columns = [])
            df_iter_base = df_iter_base.reset_index()
            df_iter_base = df_iter_base.merge(df_expvol_all.reset_index(), how = 'left', left_on = ['Date', 'Code'], right_on = ['DatePoint', 'Country'])
            df_iter_base.drop(axis = 1, labels = ['Country', 'DatePoint', 'None', 'By_Value'], inplace = True)
            df_iter_base = df_iter_base.merge(df_month_back.reset_index(), how = 'left', left_on = ['Date', 'Code'], right_on = ['Date', 'Code'])       
            df_iter_base.rename(columns = {'By_Date': 'Factor', 'Returns': 'Next Returns'}, inplace = True)            
            df_iter_base = df_iter_base.set_index(['Date', 'Code'])     
        ### Generating table for "Exponential Volatility MRI Conditional Weighted for Now / Next" factor:
        if (iter_factor == 'expvol_cond_weighted'):
            df_iter_base = pd.DataFrame(np.NaN, index = ser_returns.index, columns = [])
            df_iter_base = df_iter_base.reset_index()
            df_iter_base = df_iter_base.merge(df_expvol_all.reset_index(), how = 'left', left_on = ['Date', 'Code'], right_on = ['DatePoint', 'Country'])
            df_iter_base.drop(axis = 1, labels = ['Country', 'DatePoint', 'None', 'By_Date'], inplace = True)
            df_iter_base = df_iter_base.merge(df_month_back.reset_index(), how = 'left', left_on = ['Date', 'Code'], right_on = ['Date', 'Code'])       
            df_iter_base.rename(columns = {'By_Value': 'Factor', 'Returns': 'Next Returns'}, inplace = True)            
            df_iter_base = df_iter_base.set_index(['Date', 'Code'])     
        ### Generating table for "Exponential Volatility Surprise for Now / Next" factor:
        if (iter_factor == 'volatility_surprise'):
            df_expvol_all_full = df_expvol_all.copy()
            df_expvol_all_full['Surprise'] = -np.log(df_expvol_all_full['By_Date'] / df_expvol_all_full['By_Value'])
            df_iter_base = pd.DataFrame(np.NaN, index = ser_returns.index, columns = [])
            df_iter_base = df_iter_base.reset_index()
            df_iter_base = df_iter_base.merge(df_expvol_all_full.reset_index(), how = 'left', left_on = ['Date', 'Code'], right_on = ['DatePoint', 'Country'])
            df_iter_base.drop(axis = 1, labels = ['Country', 'DatePoint', 'None', 'By_Date', 'By_Value'], inplace = True)
            df_iter_base = df_iter_base.merge(df_month_back.reset_index(), how = 'left', left_on = ['Date', 'Code'], right_on = ['Date', 'Code'])       
            df_iter_base.rename(columns = {'Surprise': 'Factor', 'Returns': 'Next Returns'}, inplace = True)            
            df_iter_base = df_iter_base.set_index(['Date', 'Code'])                
        ### Collecting factor tables to dictinary:
        dict_factor_pairs_container[iter_factor] = df_iter_base
        print('hh_msci_factors:', 'Factor', iter_factor, 'data prepared successfully.')

    print('hh_msci_factors:', 'Container for all factors created successfully.')
    return [ser_returns, dict_factor_pairs_container]


def hh_msci_efficacy_measures(df_factor, arr_measures, market_caps_path, market_caps_key, period_quan = 999):
    """
    Version 0.01 2019-07-6
    
    FUNCTIONALITY: 
      1) Calculating efficacy measures  
      2) Calculating aggregating results for efficacy measures
    OUTPUT:
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    INPUT:
      df_factor (pd.DataFrame) - date indexed data table with factor and returns data       
      arr_measures (array) - list of string efficacy measures for factors, such as:   
          'ic_spearman' - Spearmen correlation coefficient;
          'ic_pearson' - Pearson correlation coefficient;          
          'fmb_mcap' - Fama-McBeth mcap-weighted cross-sectional regression coefficient;
          'fmb_eqw' - Fama-McBeth equal-weighted cross-sectional regression coefficient;
          'fmb_mcap_std' - Fama-McBeth mcap-weighted cross-sectional regression coefficient with standartized factor;
          'fmb_eqw_std' - Fama-McBeth equal-weighted cross-sectional regression coefficient with standartized factor; 
      market_caps_path (string) - path to the MSCI market caps HDF5 file
      market_caps_key (string) - data object key to access  MSCI market caps from HDF5 file           
      period_quan (number) - quantity of periods to step back for efficacy measures aggregating
    """
    
    import numpy as np
    import pandas as pd    
    import statsmodels.api as sm 
    from scipy import stats as ss    
    ### Expanding visibility zone for Python engine to make HH Modules seen:
    import sys 
    sys.path.append('../..')    
    ### Importing internal functions
    from HH_Modules.hh_ts import hh_simple_standartize  
    ### Initialising constants:
    arr_const_fmb_trunc = [2.5, 2]
    date_format = '%Y-%m-%d'       
    ### Preparing test market capitalization data:
    df_market_caps = pd.read_hdf(market_caps_path, market_caps_key)    
    df_market_caps.reset_index(level = 'Country', drop = True, inplace = True)
    df_market_caps.drop('INDEX', level = 'Code', inplace = True)
    df_market_caps.sort_index(level = [0, 1], inplace = True)
    ser_market_caps = df_market_caps.squeeze()
    ser_market_caps.name = 'Market Caps'
    ### Preparing measures full data vector:
    arr_index_values = [arr_measures, df_factor.index.get_level_values(0).unique().array]
    index_ser_measures_full = pd.MultiIndex.from_product(arr_index_values, names = ['Measure', 'Date'])
    ser_measures_full = pd.Series(np.NaN, index = index_ser_measures_full)    
    ### Looping efficacy measures:
    for iter_measure in arr_measures: 
        ### Cross-sectional looping:
        for iter_date in df_factor.index.get_level_values(0).unique():
            df_cross_both = df_factor.loc[iter_date].dropna(how = 'any')
            if (len(df_cross_both) == 0):
                iter_result = np.NaN
            else:       
                ### Spearmen information coefficient:
                if (iter_measure == 'ic_spearman'):
                    arr_cross_next = df_cross_both['Next Returns'].values                    
                    arr_cross_factor = df_cross_both['Factor'].values
                    iter_result = ss.spearmanr(arr_cross_factor, arr_cross_next).correlation
                ### Pearson information coefficient:                            
                if (iter_measure == 'ic_pearson'):
                    arr_cross_next = df_cross_both['Next Returns'].values                    
                    arr_cross_factor = df_cross_both['Factor'].values                    
                    iter_result = ss.pearsonr(arr_cross_factor, arr_cross_next)[0]
                ### Fama-McBeth cross-sectional regression beta coefficient (equal weighted residuals):                            
                if (iter_measure == 'fmb_eqw'):
                    arr_cross_next = df_cross_both['Next Returns'].values                                        
                    arr_cross_factor_plus_const = sm.add_constant(df_cross_both['Factor'].values)                  
                    arr_weights = np.ones(len(arr_cross_next)).tolist()
                    wls_model = sm.WLS(arr_cross_next, arr_cross_factor_plus_const, weights = arr_weights)
                    wls_results = wls_model.fit()
                    iter_result = wls_results.params[1]                  
                ### Fama-McBeth cross-sectional regression beta coefficient (equal weighted residuals and standartized factor):
                if (iter_measure == 'fmb_eqw_std'):
                    arr_cross_next = df_cross_both['Next Returns'].values                     
                    ser_weights = pd.Series(1, df_cross_both.index)                   
                    arr_cross_factor_standartized = hh_simple_standartize(df_cross_both['Factor'], ser_weights, arr_const_fmb_trunc,
                                                                          reuse_outliers = False, center_result = True)[0].values                       
                    arr_cross_factor_standartized_plus_const = sm.add_constant(arr_cross_factor_standartized)
                    arr_weights = ser_weights.values                   
                    wls_model = sm.WLS(arr_cross_next, arr_cross_factor_standartized_plus_const, weights = arr_weights)
                    wls_results = wls_model.fit()
                    iter_result = wls_results.params[1] 
                ### Fama-McBeth cross-sectional regression beta coefficient (market capitalization weighted residuals):                            
                if (iter_measure == 'fmb_mcap'):                    
                    df_cross_plus_caps = df_cross_both.join(ser_market_caps.loc['2000-01-31'], how = 'left').dropna(how = 'any')
                    arr_cross_next = df_cross_plus_caps['Next Returns'].values
                    arr_cross_factor_plus_const = sm.add_constant(df_cross_plus_caps['Factor'].values)               
                    arr_weights = df_cross_plus_caps['Market Caps'].values
                    wls_model = sm.WLS(arr_cross_next, arr_cross_factor_plus_const, weights = arr_weights)
                    wls_results = wls_model.fit()
                    iter_result = wls_results.params[1]                            
                ### Fama-McBeth cross-sectional regression beta coefficient (market capitalization weighted residuals and standartized factor):                            
                if (iter_measure == 'fmb_mcap_std'):                    
                    df_cross_plus_caps = df_cross_both.join(ser_market_caps.loc['2000-01-31'], how = 'left').dropna()
                    arr_cross_next = df_cross_plus_caps['Next Returns'].values
                    arr_cross_factor_standartized = hh_simple_standartize(df_cross_plus_caps['Factor'], df_cross_plus_caps['Market Caps'], arr_const_fmb_trunc,
                                                                          reuse_outliers = False, center_result = True)[0].values                     
                    arr_cross_factor_standartized_plus_const = sm.add_constant(arr_cross_factor_standartized) 
                    arr_weights = df_cross_plus_caps['Market Caps'].values
                    wls_model = sm.WLS(arr_cross_next, arr_cross_factor_standartized_plus_const, weights = arr_weights)
                    wls_results = wls_model.fit()
                    iter_result = wls_results.params[1]                                           
                ser_measures_full.loc[pd.IndexSlice[iter_measure, iter_date]] = iter_result  
    print('hh_msci_efficacy_measures:', 'Factor efficacy measures calculated successfully.')
    
    print('hh_msci_efficacy_measures:', 'Factor measures aggregating values calculated successfully.')
    return [ser_measures_full, ser_market_caps]