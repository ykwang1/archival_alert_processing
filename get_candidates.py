from math import ceil

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import json 
import datetime
from amplitude_cut import plot_outburst, from_database_lasair, galactic_latitude, dc_mag, batch_get_lightcurves, SIMBAD_EXCLUDES, is_excluded_simbad_class
from os import listdir
from os.path import isfile, join
DATA_PATH = '/epyc/users/ykwang/data/ac_archival/'
N = 7 #days

def convert_ts_jd(ts=None, jd=None):
    if jd == None:
        dt = datetime.datetime.strptime(ts, '%Y%m%d')
        return int(pd.Timestamp(dt).to_julian_date() + .5)
    elif ts == None:
        return pd.to_datetime(np.floor(jd), unit='D', origin='julian').strftime('%Y%m%d')
    else:
        return np.nan
    
def make_cuts(df):
    data = df.groupby('ztf_object_id').agg(np.nanmean)
    data['outburst_avg'] = np.nansum(data[['outburst_fid1', 'outburst_fid2', 'outburst_fid3']], axis=1) / 3
    data['outburst_max'] = np.nanmax(data[['outburst_fid1', 'outburst_fid2', 'outburst_fid3']], axis=1)
#     data['last_obs_jd'] = np.floor(np.nanmax(data[['last_obs_fid1', 'last_obs_fid2', 'last_obs_fid3']], axis=1))
#     data['last_obs_jd'] = data['last_obs_jd'].astype(int)
#     data['ts'] = [convert_ts_jd(jd=x) for x in data['last_obs_jd']]
    data = data.query('outburst_max == 1')
    
    max_lat = 15 # deg
    data['gal_lat'] = galactic_latitude(data['ra'], data['dec'])
    # data = data.query(f'abs(gal_lat) < {max_lat}')
    
    data = data.query('sgscore > .9')
    
    # otypes = [is_excluded_simbad_class(data.loc[ii, 'ra'], data.loc[ii, 'dec']) for ii in data.index]
    
    # data['otypes'] = otypes
    # data = data.loc[~data['otypes'].isin(SIMBAD_EXCLUDES)]
    
    # how to eliminate asteroids?? elong? don't have prev_obs data
    return data
    
def get_simbad_otypes(ras, decs):
    assert len(ras) == len(decs)
    otypes = np.array([is_excluded_simbad_class(ras[ii], decs[ii]) for ii in range(len(ras))])
    return otypes

def make_simbad_cut(df, candids, excluded_otypes, debug=False):
    pos = df.groupby('ztf_object_id')[['ra', 'dec']].mean()
    pos = pos.loc[candids]
    otypes = get_simbad_otypes(pos['ra'], pos['dec'])
    mask = np.array([True if x not in excluded_otypes else False for x in otypes])
    oids = np.array(pos.index)
    if debug==True:
        return oids, mask, otypes
    return oids[mask]

def get_candidates(df, jd, nobs=3):
    counts = pd.Series(df['ztf_object_id']).value_counts()
    oids_gte_nobs = set(counts[counts >= nobs].index)
    oids_jd = set(df.query(f'last_obs_jd == {jd}')['ztf_object_id'])
    return list(oids_jd.intersection(oids_gte_nobs))

if __name__ == "__main__":
    data_files = [f for f in listdir(DATA_PATH) if isfile(join(DATA_PATH, f))]
    csv_ts = [f.split('_')[0] for f in data_files]
    sorted_data_files = pd.Series(data_files).sort_values().reset_index(drop=True)
    sorted_ts = pd.Series(csv_ts).sort_values().reset_index(drop=True)
    dd = []
    for ts in sorted_ts[:N]:
        df = pd.read_csv(f'{DATA_PATH}{ts}_public.csv')
        print(ts, len(df))
        process = make_cuts(df)
        process['last_obs_jd'] = convert_ts_jd(ts=ts)
        process['ts'] = ts
        dd.append(process)
        print(len(process))
        
    all_data = pd.concat(dd)
    all_data.reset_index(inplace=True)
    jd = convert_ts_jd(sorted_ts[N-1])
    jd_candids = get_candidates(all_data, jd)
    oids, mask, otypes = make_simbad_cut(all_data, jd_candids, SIMBAD_EXCLUDES, debug=True)
    candids = oids[mask]
    all_candids = {sorted_ts[N-1]: list(candids)}
    
    for ii, ts in enumerate(sorted_ts):
        if ii < N:
            continue
        try:
            df = pd.read_csv(f'{DATA_PATH}{ts}_public.csv')
        except:
            all_candids[ts] =[]
            continue
        jd = convert_ts_jd(ts=ts)
        process = make_cuts(df)
        process['last_obs_jd'] = jd
        process['ts'] = ts
        process.reset_index(inplace=True)
        all_data = all_data.query(f'last_obs_jd > {jd - N}')
        all_data = pd.concat([all_data, process])
        test_candids = get_candidates(all_data, jd)
        if len(test_candids) == 0:
            all_candids[ts] = []
            continue
        try:
            oids, mask, otypes = make_simbad_cut(all_data, test_candids, SIMBAD_EXCLUDES, debug=True)
            candids = oids[mask]
        except:
            all_candids[ts] = []
            continue# import pdb; pdb.set_trace()
        print(ts, len(df), len(candids))
        all_candids[ts] = list(candids)
    with open("candids_test.json", "w") as outfile:
        json.dump(all_candids, outfile)
    
