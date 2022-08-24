from math import ceil

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import json 
import sys
from amplitude_cut import plot_outburst, from_database_lasair, galactic_latitude, dc_mag, batch_get_lightcurves, \
SIMBAD_EXCLUDES, is_excluded_simbad_class, read_avro_bytes
from get_candidates import make_simbad_cut
from os import listdir
from os.path import isfile, join


def calculate_outburst(dflc=None, ts=None, oids=None):
    if ts is not None:
        dflc = pd.read_csv(f'{LC_DIR}{ts}_public.csv')
    
    dflc = dflc.rename(columns={'ZTF_object_id':'ztf_object_id'})
    dflc['dc_mag'], dflc['dc_sigmag'], dflc['dc_mag_ulim'] = dc_mag(dflc['isdiffpos'], dflc['magnr'], dflc['magpsf'], dflc['sigmagnr'], dflc['sigmapsf'], dflc['diffmaglim'])

    grp = dflc.groupby(['fid','field','rcid'])
    impute_magnr = grp['magnr'].agg(lambda x: np.median(x[np.isfinite(x)]))
    # print(impute_magnr)
    impute_sigmagnr = grp['sigmagnr'].agg(lambda x: np.median(x[np.isfinite(x)]))
    # print(impute_sigmagnr)

    for idx, grpi in grp:
        w = np.isnan(grpi['magnr'])
        w2 = grpi[w].index
        dflc.loc[w2,'magnr'] = impute_magnr[idx]
        dflc.loc[w2,'sigmagnr'] = impute_sigmagnr[idx]
        
    dflc.reset_index(inplace=True, drop=True)
    
    if oids is None:
        oids = dflc['ztf_object_id'].unique()
    
    dflc['combined_mag_ulim'] = [dflc.loc[ii, 'dc_mag_ulim'] if pd.isna(dflc.loc[ii, 'dc_mag']) else dflc.loc[ii, 'dc_mag'] for ii in dflc.index]
    dflc['utc'] = pd.to_datetime(dflc['jd'], unit='D', origin='julian')
    
    dflc.set_index(['ztf_object_id', 'fid'], inplace=True)
    dflc = dflc.sort_values('utc').sort_index()
    
    if oids is None:
        oids = dflc['ztf_object_id']
        
    # only include candidates where faintest dc_mag < reference mag
    magnr_diffs = np.array([(dflc.loc[oid]['magnr'] - dflc.loc[oid]['dc_mag']).max() for oid in oids])
    candids = np.array(oids)[magnr_diffs > 0]
    
    # downselect candidates
    dflc_interest = dflc.loc[candids]
    dflc_interest.sort_index(inplace=True)
    
    # Calculate ewmas
    dflc_interest['ema2'] = dflc_interest.groupby(["ztf_object_id", "fid"]).apply(lambda x: x["dc_mag"].ewm(halflife='2 days', times=x['utc']).mean()).values
    dflc_interest['ema8'] = dflc_interest.groupby(["ztf_object_id", "fid"]).apply(lambda x: x["dc_mag"].ewm(halflife='8 days', times=x['utc']).mean()).values
    dflc_interest['ema28'] = dflc_interest.groupby(["ztf_object_id", "fid"]).apply(lambda x: x["dc_mag"].ewm(halflife='28 days', times=x['utc']).mean()).values

    dflc_interest['combined_ema2'] = dflc_interest.groupby(["ztf_object_id", "fid"]).apply(lambda x: x["combined_mag_ulim"].ewm(halflife='2 days', times=x['utc']).mean()).values
    dflc_interest['combined_ema8'] = dflc_interest.groupby(["ztf_object_id", "fid"]).apply(lambda x: x["combined_mag_ulim"].ewm(halflife='8 days', times=x['utc']).mean()).values
    dflc_interest['combined_ema28'] = dflc_interest.groupby(["ztf_object_id", "fid"]).apply(lambda x: x["combined_mag_ulim"].ewm(halflife='28 days', times=x['utc']).mean()).values

    ema2_m_ema8 = (dflc_interest['ema2'] - dflc_interest['ema8']) 
    ema8_m_ema28 = (dflc_interest['ema8'] - dflc_interest['ema28'])
    
    dflc_interest['outburst_metric'] = ema2_m_ema8 + ema8_m_ema28 - .3
    
    return dflc_interest

df = calculate_outburst(pd.read_csv('../data/full_dedup_lc.csv'))
df.to_csv('augmented_lcs.csv')
