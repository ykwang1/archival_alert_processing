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
from itertools import repeat
from os import listdir
from os.path import isfile, join
from amplitude_cut import plot_outburst, galactic_latitude, dc_mag, batch_get_lightcurves, SIMBAD_EXCLUDES, is_excluded_simbad_class, read_avro_bytes

from .config import ARCHIVAL_DIR, LC_SAVE_DIR, program, CANDIDS_JSON_PATH, CANDIDS_TXT_PATH, oid_csv_path, LC_FIELDS, OID_FIELDS, LC_UPDATE_FILE

SAVE_OID_FIELDS = False

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
    print(ts_dir)
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
    try:
        print(f'Total alerts: {len(tar.getmembers())} beginning...')
    except Exception as e:
        print(f"can't open {tarball_dir}", e) 
        return 0, ts_dir
        
    for ii, tarpacket in enumerate(tar.getmembers()):
#         if ii%5000 == 0:
#             print(f"{ii} messaged consumed")
        try:
            packet = read_avro_bytes(tar.extractfile(tarpacket).read())
            if packet['objectId'] in candids: 
                # print(packet['objectId'])
                processed.append(make_dataframe(packet, save_oid_fields=SAVE_OID_FIELDS))
        except Exception as e:
            # print(packet['objectId'])
            print(f'error reading an oject in {tarball_dir}', e)
            continue
    try:
        if len(processed) > 0:
            data = pd.concat(processed)
            data.to_csv(f'{LC_SAVE_DIR}{ts}_{program}.csv', index=False)
        return 1, ts_dir
    except Exception as e:
        print(f'error saving {tarball_dir}', e)
        return 0, ts_dir
    return 0, ts_dir

def main():
    pass
    
def consume_archives_parallel(json_path, program='public', n_cores=48, save=True):
    niceness = 10
    os.nice(niceness)
    p = mp.Pool(n_cores)   
#     with open('/epyc/users/ykwang/scripts/candids_test.json') as f:
#         sample = json.load(f)
    
#     full_oids = pd.Series([x for ts in sample.keys() for x in sample[ts]]).unique()
    with open(CANDIDS_TXT_PATH) as f:  
        full_oids = np.array([x.strip() for x in f.readlines()])
        
    data_files = [f for f in listdir(ARCHIVAL_DIR+'public/') if isfile(join(ARCHIVAL_DIR+'public/', f))]
    csv_ts = [f.split('public_')[-1].split('.')[0][:8] for f in data_files]
    alert_ts = list(set(csv_ts).remove(''))
    
    # ts = [k for k, v in sample.items()]
    
    statuses = p.starmap(get_dflc, zip(alerts_ts, repeat(full_oids)))
    
    with open(UPDATE_FILE, 'w') as f:
        assert len(statuses) == len(alerts_ts)
        for ii, status in enumerate(statuses):
            f.write(f"{status[0]}, {status[1]}\n")

if __name__ == '__main__':
    consume_archives_parallel(JSON_PATH)
