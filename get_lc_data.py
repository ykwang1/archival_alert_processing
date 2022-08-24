import tarfile
import fastavro
import io
import os
import json
import sqlite3
import pandas as pd
import numpy as np
import pdb
import matplotlib.pyplot as plt
import multiprocessing as mp
from itertools import repeat
import astropy.units as u
from astroquery.simbad import Simbad
from math import ceil
import datetime
from amplitude_cut import plot_outburst, galactic_latitude, dc_mag, batch_get_lightcurves, SIMBAD_EXCLUDES, is_excluded_simbad_class, read_avro_bytes

ARCHIVAL_DIR = '/epyc/data/ztf/alerts/'
SAVE_DIR = '/epyc/users/ykwang/data/ac_lc_data/'
program='public'
DATA_PATH = '/epyc/users/ykwang/data/ac_archival/'
JSON_PATH = '/epyc/users/ykwang/scripts/candids_test.json'
oid_csv_path = '/epyc/users/ykwang/scripts/object_data.csv'
SAVE_OID_FIELDS = True
LC_FIELDS = ["jd", "fid", "magpsf", "sigmapsf", "diffmaglim", "isdiffpos", "magnr", "sigmagnr", "field", "rcid"]
OID_FIELDS = ["ssdistnr", "elong", "objectidps1", "distpsnr1", "sgmag1", "srmag1", "simag1"]
UPDATE_FILE='lc_status.txt'

def make_dataframe(packet, save_oid_fields=False, repeat_obs=False):
    """Extract relevant lightcurve data from packet into pandas DataFrame."""
    df = pd.DataFrame(packet["candidate"], index=[0])
    if repeat_obs:
        df["ZTF_object_id"] = packet["objectId"]
        return df[["ZTF_object_id"] + LC_FIELDS]

    df_prv = pd.DataFrame(packet["prv_candidates"])
    df_merged = pd.concat([df, df_prv], ignore_index=True)
    df_merged["ZTF_object_id"] = packet["objectId"]
    if save_oid_fields:
        df["ZTF_object_id"] = packet["objectId"]
        df[["ZTF_object_id"] + OID_FIELDS].to_csv(oid_csv_path, mode='a', index=False, header=False)
    return df_merged[["ZTF_object_id"] + LC_FIELDS]

def get_dflc(ts, candids, program='public', save=True):
    if ts == '20180828':
        ts_dir = '.20180828'
    else:
        ts_dir = ts
    tarball_name = f'ztf_{program}_{ts_dir}.tar.gz'
    tarball_dir = ARCHIVAL_DIR + program + '/' + tarball_name
    try:
        tar = tarfile.open(tarball_dir, 'r:gz')
    except tarfile.ReadError:
        print(f'Read error reading {tarball_dir}')
        return 0, ts_dir
    except Exception as e:
        print(f'Other error reading {tarball_dir}, {e}')
        return 0, ts_dir
    
    
    processed = []
    print(f'Total alerts: {len(tar.getmembers())} beginning...')
    for ii, tarpacket in enumerate(tar.getmembers()):
#         if ii%5000 == 0:
#             print(f"{ii} messaged consumed")
        try:
            packet = read_avro_bytes(tar.extractfile(tarpacket).read())
            if packet['objectId'] in candids: 
                print(packet['objectId'])
                processed.append(make_dataframe(packet, save_oid_fields=SAVE_OID_FIELDS))
        except Exception as e:
            # print(packet['objectId'])
            print(f'error reading an oject in {tarball_dir}', e)
            continue
    if len(processed) > 0:
        data = pd.concat(processed)
        data.to_csv(f'{SAVE_DIR}{ts}_{program}.csv', index=False)
    return 1, ts_dir

def main():
    pass
    
def consume_archives_parallel(json_path, program='public', n_cores=48, save=True):
    niceness = 10
    os.nice(niceness)
    p = mp.Pool(n_cores)   
    with open('/epyc/users/ykwang/scripts/candids_test.json') as f:
        sample = json.load(f)
        
    args = [(k, v) for k, v in sample.items()]
    
    statuses = p.starmap(get_dflc, args)
    
    with open(UPDATE_FILE, 'w') as f:
        assert len(statuses) == len(args)
        for ii, status in enumerate(statuses):
            f.write(f"{status[0]}, {status[1]}\n")

if __name__ == '__main__':
    consume_archives_parallel(JSON_PATH)
