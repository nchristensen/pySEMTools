
def file_field_index(source_code: str = 'Neko_old', custom_file_names: dict = None):
    """
    Returns a dictionary with the file type and the field name for the statistics fields.
    """

    if source_code == 'Neko':

        if not isinstance(custom_file_names, type(None)):
            mean_field_name = custom_file_names.get('mean_field_name', 'mean_field.fld')
            stats_field_name = custom_file_names.get('stats_field_name', 'stats.fld')
        else:
            mean_field_name = 'mean_field.fld'
            stats_field_name = 'stats.fld'

        neko_stat_field_index = {}

        neko_stat_field_index['P'] = {'file_name' : mean_field_name, 'field_key' : 'pres'}
        neko_stat_field_index['U'] = {'file_name' : mean_field_name, 'field_key' : 'vel_0'}
        neko_stat_field_index['V'] = {'file_name' : mean_field_name, 'field_key' : 'vel_1'}
        neko_stat_field_index['W'] = {'file_name' : mean_field_name, 'field_key' : 'vel_2'}

        neko_stat_field_index['PP'] = {'file_name' : stats_field_name, 'field_key' : 'pres'}
        neko_stat_field_index['UU'] = {'file_name' : stats_field_name, 'field_key' : 'vel_0'}
        neko_stat_field_index['VV'] = {'file_name' : stats_field_name, 'field_key' : 'vel_1'}
        neko_stat_field_index['WW'] = {'file_name' : stats_field_name, 'field_key' : 'vel_2'}
        
        neko_stat_field_index['UV'] = {'file_name' : stats_field_name, 'field_key' : 'temp'}
        neko_stat_field_index['UW'] = {'file_name' : stats_field_name, 'field_key' : 'scal_0'}
        neko_stat_field_index['VW'] = {'file_name' : stats_field_name, 'field_key' : 'scal_1'}

        neko_stat_field_index['UUU'] = {'file_name' : stats_field_name, 'field_key' : 'scal_2'}
        neko_stat_field_index['VVV'] = {'file_name' : stats_field_name, 'field_key' : 'scal_3'}
        neko_stat_field_index['WWW'] = {'file_name' : stats_field_name, 'field_key' : 'scal_4'}

        neko_stat_field_index['UUV'] = {'file_name' : stats_field_name, 'field_key' : 'scal_5'}
        neko_stat_field_index['UUW'] = {'file_name' : stats_field_name, 'field_key' : 'scal_6'}
        neko_stat_field_index['UVV'] = {'file_name' : stats_field_name, 'field_key' : 'scal_7'}
        neko_stat_field_index['UVW'] = {'file_name' : stats_field_name, 'field_key' : 'scal_8'}
        neko_stat_field_index['VVW'] = {'file_name' : stats_field_name, 'field_key' : 'scal_9'}
        neko_stat_field_index['UWW'] = {'file_name' : stats_field_name, 'field_key' : 'scal_10'}
        neko_stat_field_index['VWW'] = {'file_name' : stats_field_name, 'field_key' : 'scal_11'}
        
        neko_stat_field_index['UUUU'] = {'file_name' : stats_field_name, 'field_key' : 'scal_12'}
        neko_stat_field_index['VVVV'] = {'file_name' : stats_field_name, 'field_key' : 'scal_13'}
        neko_stat_field_index['WWWW'] = {'file_name' : stats_field_name, 'field_key' : 'scal_14'}
        
        neko_stat_field_index['PPP'] = {'file_name' : stats_field_name, 'field_key' : 'scal_15'}
        neko_stat_field_index['PPPP'] = {'file_name' : stats_field_name, 'field_key' : 'scal_16'}

        neko_stat_field_index['PU'] = {'file_name' : stats_field_name, 'field_key' : 'scal_17'}
        neko_stat_field_index['PV'] = {'file_name' : stats_field_name, 'field_key' : 'scal_18'}
        neko_stat_field_index['PW'] = {'file_name' : stats_field_name, 'field_key' : 'scal_19'}

        neko_stat_field_index['PDU/DX'] = {'file_name' : stats_field_name, 'field_key' : 'scal_20'}
        neko_stat_field_index['PDU/DY'] = {'file_name' : stats_field_name, 'field_key' : 'scal_21'}
        neko_stat_field_index['PDU/DZ'] = {'file_name' : stats_field_name, 'field_key' : 'scal_22'}
        
        neko_stat_field_index['PDV/DX'] = {'file_name' : stats_field_name, 'field_key' : 'scal_23'}
        neko_stat_field_index['PDV/DY'] = {'file_name' : stats_field_name, 'field_key' : 'scal_24'}
        neko_stat_field_index['PDV/DZ'] = {'file_name' : stats_field_name, 'field_key' : 'scal_25'}
        
        neko_stat_field_index['PDW/DX'] = {'file_name' : stats_field_name, 'field_key' : 'scal_26'}
        neko_stat_field_index['PDW/DY'] = {'file_name' : stats_field_name, 'field_key' : 'scal_27'}
        neko_stat_field_index['PDW/DZ'] = {'file_name' : stats_field_name, 'field_key' : 'scal_28'}

        neko_stat_field_index['E11'] = {'file_name' : stats_field_name, 'field_key' : 'scal_29'}
        neko_stat_field_index['E22'] = {'file_name' : stats_field_name, 'field_key' : 'scal_30'}
        neko_stat_field_index['E33'] = {'file_name' : stats_field_name, 'field_key' : 'scal_31'}
        neko_stat_field_index['E12'] = {'file_name' : stats_field_name, 'field_key' : 'scal_32'}
        neko_stat_field_index['E13'] = {'file_name' : stats_field_name, 'field_key' : 'scal_33'}
        neko_stat_field_index['E23'] = {'file_name' : stats_field_name, 'field_key' : 'scal_34'}


        stat_field_index = neko_stat_field_index
    
    else:
        raise ValueError('The source code is not supported. Please use "Neko" as the source code.')

    return stat_field_index
