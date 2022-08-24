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
import wget
from astropy.coordinates import SkyCoord, match_coordinates_sky
import astropy.units as u
from astroquery.simbad import Simbad
import lasair
from math import ceil

SIMBAD_EXCLUDES = ['G?', 'SC?', 'C?G', 'Gr?', 'As?', 'Y*?', 'pr?', 'TT?', 'Mi?', 'SCG', 'ClG',
'GrG', 'CGG', 'PaG', 'IG', 'Y*O', 'pr*', 'TT*', 'Or*', 'FU*', 'BY*', 'RS*',
'Pu*', 'RR*', 'Ce*', 'dS*', 'RV*', 'WV*', 'bC*', 'cC*', 'gD*', 'LP*', 'Mi*',
'SN*', 'su*', 'G', 'PoG', 'GiC', 'BiC', 'GiG', 'GiP', 'HzG', 'ALS', 'LyA',
'DLA', 'mAL', 'LLS', 'BAL', 'rG', 'H2G', 'LSB', 'AG?', 'Q?', 'Bz?', 'BL?',
'EmG', 'SBG', 'bCG', 'LeI', 'LeG', 'LeQ', 'AGN', 'LIN', 'SyG', 'Sy1', 'Sy2',
'Bla', 'BLL', 'QSO'] + ['HII',  'No*', 'MoC', 'Cld', 'HH', 'Ae*'] # Additional otypes from 210630 meeting
SAVE=True # save as you go
UPDATE_FILE = 'consumed_files.txt' 

customSimbad = Simbad()
customSimbad.add_votable_fields("otype(3)")
customSimbad.add_votable_fields("otypes(3)")  # returns a '|' separated list of all the otypes

def query_simbad(ra, dec):
    sc = SkyCoord(ra, dec, unit=u.degree)
    result_table = customSimbad.query_region(sc, radius=2 * u.arcsecond, cache=False)
    return result_table


def is_excluded_simbad_class(ra, dec):
    """Is the object in Simbad, with object types we reject (e.g., AGN)?
    """
    try:
        result_table = query_simbad(ra, dec)
        if result_table is None:
            # print(f"Source not found in Simbad")
            return None  # not in Simbad
        else:
            # for now let's just take the closest match
            otype = result_table['OTYPE_3'][0].decode("utf-8")
            simbad_id = result_table['MAIN_ID'][0].decode("utf-8")
            if otype in SIMBAD_EXCLUDES:
                # print(f"Source found in Simbad as {simbad_id} ({otype}); omitting")
                return otype
            else:
                # print(f"Source found in Simbad as {simbad_id} ({otype}); saving")
                return otype

    except Exception as e:
        # if this doesn't work, record the exception and continue
        print("Error querying Simbad:", e)
        return None

def batch_get_lightcurves(ztf_object_list):
    LASAIR_TOKEN = '4b762569bb349bd8d60f1bc7da3f39dbfaefff9a'#'8d418fb07071223eb94c94a507186766c49c0049'
    L = lasair.lasair_client(LASAIR_TOKEN)
    dfs = []
    n_split = ceil(len(ztf_object_list) / 50)
    ztf_object_list_blocked = np.array_split(ztf_object_list, n_split)
    for object_list in ztf_object_list_blocked:
        c = L.lightcurves(object_list)
        for ii, ztf_object_id in enumerate(object_list):
            temp = pd.DataFrame(c[ii])
            temp.sort_values('jd', inplace=True)
            temp['magnr'] = temp['magnr'].fillna(method='ffill')
            temp['magnr'] = temp['magnr'].fillna(method='bfill')
            temp['dc_mag'], temp['dc_sigmag'], temp['dc_mag_ulim'] = dc_mag(temp['isdiffpos'], 
                                                                            temp['magnr'], 
                                                                            temp['magpsf'], 
                                                                            temp['sigmagnr'], 
                                                                            temp['sigmapsf'], 
                                                                            temp['diffmaglim'])
            temp['ztf_object_id'] = object_list[ii]
            dfs.append(temp)
    return pd.concat(dfs)
    
    
def plot_outburst(d, x, Tmin=0, Tmax=0, min=0, max=0):
    fids={1: 'green', 2:'red', 3:'pink'}
    for f in d.index.unique():
        tlist = np.array(d.loc[f]['jd'])
        mags  = np.array(d.loc[f]['dc_mag'])
        f02   = np.array(d.loc[f]['combined_ema2'])
        f08   = np.array(d.loc[f]['combined_ema8'])
        f28   = np.array(d.loc[f]['combined_ema28'])
        plt.scatter(tlist, mags, marker='o', color=fids[f])
        plt.plot(tlist, f02, color=fids[f], alpha=.4)
        plt.plot(tlist, f08, color=fids[f], alpha=.6)
        plt.plot(tlist, f28, color=fids[f], alpha=.8)
        nobs = len(d.loc[[f],:])
        if nobs == 1:
            continue
        for i in range(nobs):
            p = -mags[i]
            q = -f02[i]
            r = -f08[i]
            s = -f28[i]
            if q > r and r > s + 0.2:
                t = tlist[i]
                m = mags[i]
                plt.plot([t], [m], marker='o', markersize=10, color=fids[f])

        plt.xlabel('MJD')
        plt.ylabel('%s magnitude' % f)
        plt.title('%s' % (x))
#         if min > 0:
#             plt.axis(ymin=max, ymax=min)
#         if Tmin > 0:
#             plt.axis(xmin=Tmin, xmax=Tmax)
    plt.gca().invert_yaxis()
    plt.show()
    
def from_database_lasair(objectId):
    file = '%s.json' % objectId
    try:
        o = json.loads(open(file).read())
    except:
        url = 'https://lasair.roe.ac.uk/object/%s/json/' % objectId
        wget.download(url, out=file)
        o = json.loads(open(file).read())
    print (o['objectId'])
    clist = o['candidates']

    tlist = {'g':[], 'r':[]}
    mags  = {'g':[], 'r':[]}
    f02   = {'g':[], 'r':[]}
    f08   = {'g':[], 'r':[]}
    f28   = {'g':[], 'r':[]}
        
    for c in clist:
        if 'candid' in c:
            if c['fid'] == 1: f = 'g'
            else:             f = 'r'
            tlist[f].append(c['mjd'])
            mags[f].append(c['dc_mag'])
            f02[f].append(c['dc_mag_%s02'%f])
            f08[f].append(c['dc_mag_%s08'%f])
            f28[f].append(c['dc_mag_%s28'%f])
            
    return {'objectId':objectId, 'tlist':tlist, 'mags':mags, 'f02':f02, 'f08':f08, 'f28':f28}


ARCHIVAL_DIR = '/epyc/data/ztf/alerts/'
SAVE_DIR = '~/notebooks/data/ac_archival/'
program='public'

def make_dataframe(packet):
    """Extract relevant lightcurve data from packet into pandas DataFrame."""
    df = pd.DataFrame(packet["candidate"], index=[0])
    fid = packet['candidate']['fid']
    df_prv = pd.DataFrame(packet["prv_candidates"])
    df_merged = pd.concat([df_prv, df], ignore_index=True)
    df = df.query(f'fid == {fid}')
    df_merged["ztf_object_id"] = packet["objectId"]
    for col in ["magpsf", "sigmapsf", "diffmaglim", "magnr", "sigmagnr"]:
        df_merged[col] = df_merged[col].astype(float)
    df_merged["utc"] = pd.to_datetime(df_merged['jd'], unit='D', origin='julian')
    return df_merged[["ztf_object_id", "jd", "utc", "magpsf", "sigmapsf", "diffmaglim",
                      "isdiffpos", "magnr", "sigmagnr"]]

def EWMA_recursive(obs, last_obs, mag, last_ewma, tau=8):
    f = np.exp(-(obs - last_obs) / tau)
    return f * last_ewma + (1 - f) * mag

def EWMA_init(packet, combine=True, tau=8, tau_list=None, debug=False):
    '''
    Input:
        packet
    '''
    df = make_dataframe(packet)
    # df = df.dropna(subset=['magnr'])
    df['dc_mag'], df['dc_sigmags'], df['dc_mag_ulim'] = dc_mag(df['isdiffpos'], df['magnr'], 
                                                                df['magpsf'], df['sigmagnr'], 
                                                                df['sigmapsf'], df['diffmaglim'])
    df['combined_mag_ulim'] = [df.loc[ii, 'dc_mag_ulim'] 
                               if pd.isna(df.loc[ii, 'dc_mag']) 
                               else df.loc[ii, 'dc_mag'] 
                               for ii in df.index]
    
    df.sort_values('utc', inplace=True)
    
    if tau_list is not None:
        for t in tau_list:
            df[f'ewma{t}'] = df['combined_mag_ulim'].ewm(halflife=f'{t} days', times=df['utc']).mean().values
        return [df[f'ewma{t}'].values[-1] for t in tau_list]
    
    df['ewma'] = df['combined_mag_ulim'].ewm(halflife=f'{tau} days', times=df['utc']).mean().values
    if debug:
        return df
    return df['ewma'].values[-1]

def dc_mag(isdiffpos, magnr, magpsf, sigmagnr, sigmapsf, diffmaglim=None):
    sign = 2 * ((isdiffpos == 't') | (isdiffpos == '1')) - 1
    u = 10**(-0.4*magnr) + sign * 10**(-0.4*magpsf)
    
#     if np.sum(np.asarray(u)<=0) > 0:
#         print(isdiffpos)
#         print('negative log')
    dc_mag = -2.5 * np.log10(u)
    dc_sigmag = np.sqrt(
    (10**(-0.4*magnr)* sigmagnr) **2. + 
    (10**(-0.4*magpsf) * sigmapsf)**2.) / u
    
    dc_mag_ulim = -2.5 * np.log10(10**(-0.4*magnr) + 10**(-0.4*diffmaglim))
#     dc_mag_llim = -2.5 * np.log10(10**(-0.4*magnr) - 10**(-0.4*diffmaglim))
    
    return dc_mag, dc_sigmag, dc_mag_ulim

def process_alert(packet, prev_obs=None):
    """
    Takes an alert and extracts the relevant information to save
    
    Input:
        packet: json in the ZTF avro schema
        prev_obs: pd.Series with the previous datum for this source
        
    Output:
        List with the attributes in features to save
            """
    
    features_to_save = ['ztf_object_id', 'ra', 'dec', 'fid',
                        'last_obs_fid1', 'ewma2_fid1', 'ewma8_fid1', 'ewma28_fid1', 
                        'metric_fid1', 'outburst_fid1', 'outburst_time_fid1', 
                        'last_obs_fid2', 'ewma2_fid2', 'ewma8_fid2', 'ewma28_fid2', 
                        'metric_fid2', 'outburst_fid2', 'outburst_time_fid2', 
                        'last_obs_fid3', 'ewma2_fid3', 'ewma8_fid3', 'ewma28_fid3', 
                        'metric_fid3', 'outburst_fid3', 'outburst_time_fid3', 
                        'SIMBAD_otype', 'xray_name', 'interest_flag', 'sgscore', 'min_mag']
    packet_data = pd.Series([np.nan] * len(features_to_save), index = features_to_save)
    
    # Extract relevant parts of observation
    fid = packet['candidate']['fid']
    packet_data['fid'] = fid
    
    if prev_obs is not None:
        packet_data = prev_obs.copy()
    
    else:
        packet_data['ztf_object_id'] = packet['objectId']
        packet_data['ra'] = packet['candidate']['ra']
        packet_data['dec'] = packet['candidate']['dec']
        packet_data['sgscore'] = packet['candidate']['sgscore1']
        packet_data[f'outburst_fid{fid}'] = 0
    
        
    packet_data[f'last_obs_fid{fid}'] = packet['candidate']['jd']
    
            
    # Calculate DC magnitude
    if not np.isnan(packet['candidate']['magnr']):
        mag, sigmag, _ = dc_mag(packet['candidate']['isdiffpos'], packet['candidate']['magnr'], 
                             packet['candidate']['magpsf'], packet['candidate']['sigmagnr'], 
                             packet['candidate']['sigmapsf'], packet['candidate']['diffmaglim'])
    else:
        mag = diffmaglim
    # If previously observed, update ewma value
    if prev_obs is not None:
        obs_last_30d = ((packet['candidate']['jd'] - prev_obs[f'ewma2_fid{fid}']) < 30)
        if not np.isnan(prev_obs[f'last_obs_fid{fid}']) and obs_last_30d:


            
            packet_data[f'ewma2_fid{fid}'] = EWMA_recursive(packet_data[f'last_obs_fid{fid}'], 
                                                           prev_obs[f'last_obs_fid{fid}'],
                                                           mag, prev_obs[f'ewma_fid{fid}'], tau=2)
            packet_data[f'ewma8_fid{fid}'] = EWMA_recursive(packet_data[f'last_obs_fid{fid}'], 
                                                           prev_obs[f'last_obs_fid{fid}'],
                                                           mag, prev_obs[f'ewma_fid{fid}'])
            packet_data[f'ewma28_fid{fid}'] = EWMA_recursive(packet_data[f'last_obs_fid{fid}'], 
                                                           prev_obs[f'last_obs_fid{fid}'],
                                                           mag, prev_obs[f'ewma_fid{fid}'], tau=28)

        else:
            ewmas = EWMA_init(packet, tau_list=[2, 8, 28])
            packet_data[f'ewma2_fid{fid}'] = ewmas[0] # EWMA_init(packet, tau=2)
            packet_data[f'ewma8_fid{fid}'] = ewmas[1] # EWMA_init(packet)
            packet_data[f'ewma28_fid{fid}'] = ewmas[2] # EWMA_init(packet, tau=28)
    # If not previously observed, or no obs in last 30 days, recalculate EWMA
    else:
        ewmas = EWMA_init(packet, tau_list=[2, 8, 28])
        packet_data[f'ewma2_fid{fid}'] = ewmas[0] # EWMA_init(packet, tau=2)
        packet_data[f'ewma8_fid{fid}'] = ewmas[1] # EWMA_init(packet)
        packet_data[f'ewma28_fid{fid}'] = ewmas[2] # EWMA_init(packet, tau=28)
        
    packet_data[f'metric_fid{fid}'] = mag
#     # update minimum mag if relevant
#     if (mag > packet_data['min_mag']) or np.isnan(packet_data['min_mag']):
#         packet_data['min_mag'] = mag

    # determine if obj in outburst
    in_outburst = True if (((packet_data[f'ewma2_fid{fid}'] - packet_data[f'ewma8_fid{fid}']) < 0) and
                           ((packet_data[f'ewma8_fid{fid}'] - packet_data[f'ewma28_fid{fid}']) < -.3)) else False
    if not in_outburst:
        packet_data[f'outburst_fid{fid}'] = 0
        packet_data['outburst_start'] = np.nan
    else:
        if packet_data[f'outburst_fid{fid}'] == 0:
            packet_data[f'outburst_time_fid{fid}'] = packet['candidate']['jd']
        packet_data[f'outburst_fid{fid}'] += 1
    # update time in outburst if relevant
    
    return packet_data    

def read_avro_bytes(buf):
    """Reads a single packet from an avro file stored with schema on disk."""
    with io.BytesIO(buf) as f:
        freader = fastavro.reader(f)
        for packet in freader:
            return packet

def consume_one_night(file_path, program='public', n_alerts=None, save=SAVE):
#     tarball_name = f'ztf_{program}_{timestamp}.tar.gz'
#     tarball_dir = ARCHIVAL_DIR + program + '/' + tarball_name
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
        
    for ii, tarpacket in enumerate(tar.getmembers()[:n_alerts]):
#         if ii%1000 == 0:
#             print(f"{ii} messaged consumed")
        try:
            packet = read_avro_bytes(tar.extractfile(tarpacket).read())
            processed.append(process_alert(packet, None))
        except:
            # print(packet['objectId'])
            print(f'error reading an oject in {tarball_dir}')
            continue
    
    try:
        data = pd.DataFrame(processed)
        data = data.sort_values('ztf_object_id').sort_values('last_obs_fid3').sort_values('last_obs_fid2').sort_values('last_obs_fid1')
        # data.to_csv('test_night_20191202.csv', index=False)
        data = data.drop_duplicates(subset=['ztf_object_id', 'fid'], keep='last')
    except Exception as e:
        print(f'Problem manipulating df {e}') 
    if save:
        update_file = 'complete_parallel.txt'
        ts = file_path.split('_')[-1].split('.tar')[0]
        data.to_csv(f'{SAVE_DIR}{ts}_{program}.csv', index=False)
        with open(update_file, 'a') as g:
            g.write(f'{file_path}')
        return (0, file_path)
    return data

# def process_alert_parallel(tarpacket, timestamp):
#     tarball_name = f'ztf_{program}_{timestamp}.tar.gz'
#     tarball_dir = ARCHIVAL_DIR + program + '/' + tarball_name
#     tar = tarfile.open(tarball_dir, 'r:gz')
#     packet = read_avro_bytes(tar.extractfile(tarpacket).read())
#     tar.close()
#     return process_alert(packet)

# def consume_one_night_parallel(timestamp, program='public', n_cores=8, n_alerts=None):
#     tarball_name = f'ztf_{program}_{timestamp}.tar.gz'
#     tarball_dir = ARCHIVAL_DIR + program + '/' + tarball_name

#     tar = tarfile.open(tarball_dir, 'r:gz')

#     if n_alerts is None:
#         n_alerts = len(tar.getmembers())
#     p = mp.Pool(n_cores)
#     tarpackets =  tar.getmembers()[:n_alerts]
#     processed = p.starmap(process_alert_parallel, zip(tarpackets, repeat(timestamp)))

#     data = pd.DataFrame(processed)
#     data = data.sort_values('ztf_object_id').sort_values('last_obs_fid3').sort_values('last_obs_fid2').sort_values('last_obs_fid1')
#     # data.to_csv('test_night_20191202.csv', index=False)
#     data = data.drop_duplicates(subset=['ztf_object_id', 'fid'], keep='last')
#     return data

# def consume_archives(timestamp_list, program='public', save=True):
#     update_file = 'complete.txt'
#     with open(timestamp_list) as f:
#         timestamps = f.readlines()
#         for ts in timestamps:
#             df = consume_one_night(str(ts).strip())
#             df.to_csv(f'{SAVE_DIR}{ts}_{program}.csv', index=False)
#             with open(update_file, 'a') as g:
#                 g.write(f'{ts} written\n')

def consume_archives_parallel_blocked(fp_list, program='public', n_cores=8, save=True):
    p = mp.Pool(n_cores)   
    update_file = UPDATE_FILE
    # block file paths into blocks of len n_cores 
    if len(fp_list) > n_cores:
        n_split = int(len(fp_list)/(2 * n_cores))
        fp_list_blocked = np.array_split(fp_list, n_split)
    else:
        fp_list_blocked = [fp_list]
    
    for fps in fp_list_blocked:
        results = p.map(consume_one_night, fps)
#         for ii, fp in enumerate(fps):
#             ts = fp.split('_')[-1].split('.tar')[0]
#             df = results[ii]
#             df.to_csv(f'{SAVE_DIR}{ts}_{program}.csv', index=False)
#             with open(update_file, 'a') as g:
#                 g.write(f'{fp}\n')

def consume_archives_parallel(fp_list, program='public', n_cores=8, save=True):
    p = mp.Pool(n_cores)   
    update_file = UPDATE_FILE    
    results = p.map(consume_one_night, fp_list)
    with open(update_file, 'w') as f:
        assert len(results) == len(fp_list)
        for ii, status in enumerate(results):
            f.write(f"{status}, {fp_list[ii]}\n")


def galactic_latitude(ra, dec):
    # l_ref = 33.012 # deg
    ra_ref = 282.25 # deg
    g = 62.6 # deg 
    b =  np.arcsin(np.sin(np.deg2rad(dec)) * np.cos(np.deg2rad(g)) - \
                   np.cos(np.deg2rad(dec)) * np.sin(np.deg2rad(g)) * np.sin(np.deg2rad(ra) - np.deg2rad(ra_ref)))
    return np.rad2deg(b)
     
    
def consume_one_night_kafka(file_path, program='public', n_alerts=None, save=SAVE):
    parser = argparse.ArgumentParser()
    parser.add_argument("date", type=str, help="UTC date as YYMMDD")
    parser.add_argument("program_id", type=int, help="Program ID (1 or 2)")

    args = parser.parse_args()

    if len(args.date) != 6:
        raise ValueError(f"Date must be specified as YYMMDD.  Provided {args.date}")

    if args.program_id not in [1, 2]:
        raise ValueError(f"Program id must be 1 or 2.  Provided {args.program_id}")

    kafka_topic = f"ztf_20{args.date}_programid{args.program_id}"
    kafka_server = "partnership.alerts.ztf.uw.edu:9092"

    now = datetime.datetime.now().strftime("%d%m%y_%H%M%S")
    LOGGING["handlers"]["logfile"]["filename"] = f"{BASE_DIR}/../logs/{kafka_topic}_{now}.log"
    logging.config.dictConfig(LOGGING)

    logging.info(f"Args parsed and validated: {args.date}, {args.program_id}")

    logging.info(f"Connecting to Kafka topic {kafka_topic}")

    consumer = KafkaConsumer(
        kafka_topic,
        bootstrap_servers=kafka_server,
        auto_offset_reset="earliest",
        value_deserializer=read_avro_bytes,
        group_id=f"{GROUP_ID_PREFIX}catch_up{args.suffix}",
        consumer_timeout_ms=KAFKA_TIMEOUT) # ~2 hour timeout
    # Get cluster layout and join group `my-group`
    tstart = time.perf_counter()
    tbatch = tstart
    i = 0
    nmod = 1000
    logging.info('begin ingesting messages')
    processed = []

    for msg in consumer:
        i += 1
        if i % nmod == 0:
            elapsed = time.perf_counter() - tstart
            logging.info(f'Consumed {i} messages in {elapsed:.1f} sec ({i / elapsed:.1f} messages/s)')
        try:
            packet = msg.value
            processed.append(process_alert(packet, None))
        except Exception as e:
            logging.exception(e)

    try:
        data = pd.DataFrame(processed)
        data = data.sort_values('ztf_object_id').sort_values('last_obs_fid3').sort_values('last_obs_fid2').sort_values('last_obs_fid1')
        # data.to_csv('test_night_20191202.csv', index=False)
        data = data.drop_duplicates(subset=['ztf_object_id', 'fid'], keep='last')
    except Exception as e:
        logging.info(f'Problem manipulating df {e}') 
    if save:
        update_file = UPDATE_FILE
        ts = file_path.split('_')[-1].split('.tar')[0]
        data.to_csv(f'{SAVE_DIR}{ts}_{program}.csv', index=False)
        with open(update_file, 'a') as g:
            g.write(f'file_path')
        return (0, file_path)
    return data
    
if __name__ == "__main__":
#     times = ['20180723', '20180724', '20180725', '20180726', '20180727', '20180728', '20180729', '20180730']
    # consume_archives('data/test_timestamps.txt')
    print('starting...')
    import time
    s = time.time()
    with open('alert_archive_fps_public_remaining.txt', 'r') as f:
        fps = f.readlines()
    # times = [x.strip() for x in ts]
    print(f'{len(fps)} days to process')
    niceness = 10
    os.nice(niceness)
    print(f'nice value: {niceness}')
    consume_archives_parallel(fps, n_cores=3)
#     import time
#     s = time.time()
#     print('starting...')
#     data = consume_one_night('20180601')
    e = time.time()
#     # print(e - s)
#     data = consume_one_night_parallel('20180601', n_cores=16)
    print(e-s)
#     print(time.time() - e)
    
