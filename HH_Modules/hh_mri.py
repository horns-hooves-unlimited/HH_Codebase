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