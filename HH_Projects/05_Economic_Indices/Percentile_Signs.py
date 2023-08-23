
### LOOPER FOR BETA LEVEL PERCENTILE FACTOR

path_market_risk_source_hdf = 'Data_Files/Source_Files/market_risk_source.h5'
index_ret_USD_key = 'index_ret_USD_key'
monthly_ret_key = 'monthly_ret_key'
gri_level_perc_key = 'gri_level_perc_key'
ser_index_ret_USD = pd.read_hdf(path_market_risk_source_hdf, index_ret_USD_key)
ser_monthly_ret_USD = get_returns_from_index(ser_index_ret_USD, ma_wnd = 5, day_period = 21)
#ser_monthly_ret_USD.to_hdf(path_market_risk_source_hdf, monthly_ret_key, mode = 'a', format = 'table')
ser_perc_gri = pd.read_hdf(path_market_risk_source_hdf, gri_level_perc_key) ### THE ONLY CODE DIFFERENCE BETWEEN THE BETA TIMiNG FACTORS CALCULATION
market_membership_key = 'market_membership_key'
ser_market_membership = pd.read_hdf(path_market_risk_source_hdf, market_membership_key)
date_range_test = ser_market_membership.sort_index(level = 1).index.get_level_values(1).unique()
#date_range_test = ser_market_membership.sort_index(level = 1).index.get_level_values(1).unique()[155 : 156]
arr_beta_factor = []
iter_counter = 0
tumbler_to_minus = 0.60
tumbler_to_plus = 0.40
for iter_date in date_range_test[-10 : ]:
    ### Sign defining:
    ser_beta_signs = pd.Series(np.NaN, index = ser_perc_gri[ : iter_date].index)
    ser_beta_signs.iloc[0] = 1
    for signs_date in ser_beta_signs.index:
        if ser_beta_signs.index.get_loc(signs_date) > 0:
            if (ser_beta_signs.loc[signs_date - pd.offsets.BusinessDay()] == 1):
                if (ser_perc_gri[signs_date] > tumbler_to_minus):
                    ser_beta_signs.loc[signs_date] = -1
                else:
                    ser_beta_signs.loc[signs_date] = 1
            else:
                if (ser_perc_gri[signs_date] < tumbler_to_plus):
                    ser_beta_signs.loc[signs_date] = 1
                else:
                    ser_beta_signs.loc[signs_date] = -1             
    
    arr_beta_factor.append(get_beta_factor(iter_date) * ser_beta_signs[iter_date])
    
    iter_counter = iter_counter + 1    
    if ((iter_counter // 10) == (iter_counter / 10)):
        print('Progress printout', iter_counter, '/', iter_date)
        
ser_beta_factor = pd.concat(arr_beta_factor)