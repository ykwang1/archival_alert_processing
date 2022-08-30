import io
import gzip
import tarfile
import time
import datetime
import argparse
import logging
from copy import deepcopy
import numpy as np
import pandas as pd
import fastavro
from astropy.coordinates import SkyCoord, match_coordinates_sky
import astropy.units as u
from astropy.io import fits
import os
import multiprocessing as mp
from itertools import repeat
from astroquery.simbad import Simbad
from config import ARCHIVAL_DIR, XMATCH_UPDATE_FILE, ALERT_PROC_N_CORES, XMATCH_SAVE_DIR, FPS_TO_READ, CATALOG_DIR

SAVE=True

SIGMA_TO_95pctCL = 1.95996

def read_avro_file(fname):
    """Reads a single packet from an avro file stored with schema on disk."""
    with open(fname, "rb") as f:
        freader = fastavro.reader(f)
        for packet in freader:
            return packet

def read_avro_bytes(buf):
    """Reads a single packet from an avro file stored with schema on disk."""
    with io.BytesIO(buf) as f:
        freader = fastavro.reader(f)
        for packet in freader:
            return packet

def get_candidate_info(packet):
    return {"ra": packet["candidate"]["ra"], "dec": packet["candidate"]["dec"],
            "object_id": packet["objectId"], "magnr": packet["candidate"]["magnr"]}

def load_xray():
    # open combined xray catalog
    dfx = pd.read_csv(CATALOG_DIR)
    xray_skycoord = SkyCoord(ra=dfx.RA, dec=dfx.DEC,
                              frame="icrs", unit=(u.deg))
    return dfx[["xray_name", "RA", "DEC", "err_pos_arcsec"]], xray_skycoord


def not_moving_object(packet):
    """Check if there are > 2 detections separated by at least 30 minutes.
    Parameters:
       avro_packet: dictionary
          Extracted data from the avro file
    Return:
        real: boolean
            True if there is another detection > 30 minutes from the triggering detection
    """

    date0 = float(packet['candidate']['jd'])
    if packet['prv_candidates'] is None:
        return False
    if len(packet['prv_candidates']) == 0:
        return False

    for prv_candidate in packet['prv_candidates']:
        if prv_candidate['candid'] is not None:
            date1 = float(prv_candidate['jd'])
            diff_date = date0 - date1
            if diff_date < (0.5 / 24):
                continue
            else:
                return True

    return False



def ztf_rosat_crossmatch(ztf_source, xray_skycoord, dfx):
    """
    Cross match ZTF and ROSAT data using astropy.coordinates.SkyCoord
    Parameters:
                ztf_source: dict
                    {'ra': float (degrees), 'dec': float (degrees),
                    'object_id': string, 'candid': int}
                xray_skycoord: astropy.coordinates.SkyCoord
                    ROSAT catalog in astropy.coordinates.SkyCoord
                dfx: pandas.DataFrame
                    slim ROSAT catalog
    Return:
            matched_source: dict or None
                if a source matches, return
                    {'ra': float (degrees), 'dec': float (degrees),
                    'source_name': string, 'sep2d': float (arcsec)}
                else None
    """
    try:
        # Input avro data ra and dec in SkyCoords
        avro_skycoord = SkyCoord(ra=ztf_source["ra"], dec=ztf_source["dec"],
                                 frame="icrs", unit=(u.deg))

        # Finds the nearest ROSAT source's coordinates to the avro files ra[deg] and dec[deg]
        match_idx, match_sep2d, _ = avro_skycoord.match_to_catalog_sky(xray_skycoord)

        match_row = dfx.iloc[match_idx]

        matched = match_sep2d[0] <= match_row["err_pos_arcsec"] * u.arcsecond

        match_result = {"match_name": match_row["xray_name"],
                        "match_ra": match_row["RA"],
                        "match_dec": match_row["DEC"],
                        "match_err_pos": match_row["err_pos_arcsec"],
                        "match_sep": match_sep2d[0].to(u.arcsecond).value}

        if matched:
            logging.info(
                f"{ztf_source['object_id']} ({avro_skycoord.to_string('hmsdms')}; {ztf_source['candid']}) matched {match_result['match_name']} ({match_result['match_sep']:.2f} arcsec away)")
            return match_result

        else:
            return None
    except Exception as e:
        return None



def process_alert(packet, xray_skycoord, dfx):
    """Examine packet for matches in the ROSAT database. Save object to database if match found"""
    rb_key = "drb" if "drb" in packet["candidate"].keys() else 'rb'
    if packet["candidate"][rb_key] < 0.8:  # if packet real/bogus score is low, ignore
        return
    # # Not a solar system object (or no known obj within 5")
    if not((packet["candidate"]['ssdistnr'] is None) or (packet["candidate"]['ssdistnr'] < 0) or (packet["candidate"]['ssdistnr'] > 5)):
        return 

    ztf_source = get_candidate_info(packet)
    ztf_source['Xray_source'] = ztf_rosat_crossmatch(ztf_source, xray_skycoord, dfx)
    
    return ztf_source


def consume_one_night(file_path, program='public', n_alerts=None, save=SAVE):
    tarball_dir = file_path.strip()
    try:
        tar = tarfile.open(tarball_dir, 'r:gz')
    except tarfile.ReadError:
        print(f'Read error reading {tarball_dir}')
        if save:
            return (0, file_path)
        return pd.DataFrame([])
    except Exception as e:
        print(f'Other error reading {tarball_dir}, {e}')
        if save:
            return (0, file_path)
        return pd.DataFrame([])
    processed = []
    
    if n_alerts is None:
        try:
            n_alerts = len(tar.getmembers())
        except Exception as e:
            print(f'error getting members of {tarball_dir}, {e}')
            if save:
                return (0, file_path)
            return pd.DataFrame([]) 
    try:
        dfx, xray_skycoord = load_xray()
    except:
        print(f"error reading xray catalog")
        if save:
            return (0, file_path)
        return pd.DataFrame([])    
    for ii, tarpacket in enumerate(tar.getmembers()[:n_alerts]):
        if ii%1000 == 0:
            print(f"{ii} messaged consumed")
        try:
            packet = read_avro_bytes(tar.extractfile(tarpacket).read())
            
        except:
            # print(packet['objectId'])
            print(f'error reading an object in {tarball_dir}')
            continue
        try:
            processed.append(process_alert(packet, xray_skycoord, dfx))
        except Exception as e:
            print(f'error an object in {tarball_dir}, {e}')
            continue            
    
    try:
        data = pd.DataFrame(processed)
        # data.to_csv('test_night_20191202.csv', index=False)
    except Exception as e:
        print(f'Problem manipulating df {e}') 
    if save:
        update_file = 'complete_parallel.txt'
        ts = file_path.split('_')[-1].split('.tar')[0]
        data.to_csv(f'{XMATCH_SAVE_DIR}{ts}_{program}.csv', index=False)
        with open(update_file, 'a') as g:
            g.write(f'{file_path}')
        return (0, file_path)
    return data

def xmatch_archives_parallel(fp_list, program='public', n_cores=8, save=True):
    p = mp.Pool(n_cores)   
    update_file = XMATCH_UPDATE_FILE    
    results = p.map(consume_one_night, fp_list)
    with open(update_file, 'w') as f:
        assert len(results) == len(fp_list)
        for ii, status in enumerate(results):
            f.write(f"{status}, {fp_list[ii]}\n")


if __name__ == "__main__":
#     times = ['20180723', '20180724', '20180725', '20180726', '20180727', '20180728', '20180729', '20180730']
    print('starting...')
    import time
    s = time.time()
    with open(FPS_TO_READ, 'r') as f:
        fps = f.readlines()
    print(f'{len(fps)} days to process')
    niceness = 10
    os.nice(niceness)
    print(f'nice value: {niceness}')
    xmatch_archives_parallel(fps, n_cores=ALERT_PROC_N_CORES)
    e = time.time()

    print(e-s)

    

