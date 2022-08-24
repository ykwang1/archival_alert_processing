ARCHIVAL_DIR = '/epyc/data/ztf/alerts/'
ALERT_SAVE_DIR = '/epyc/users/ykwang/data/ac_pipeline/alert_data'
LC_SAVE_DIR = '/epyc/users/ykwang/data/ac_pipeline/lc_data/'
# FPS_TO_READ = '/epyc/users/ykwang/data/ac_pipeline/alert_archive_fps_public_remaining.txt'
FPS_TO_READ = '/epyc/users/ykwang/data/ac_pipeline/known_xrb_fps.txt'
program='public'

SIMBAD_EXCLUDES = ['G?', 'SC?', 'C?G', 'Gr?', 'As?', 'Y*?', 'pr?', 'TT?', 'Mi?', 'SCG', 'ClG',
'GrG', 'CGG', 'PaG', 'IG', 'Y*O', 'pr*', 'TT*', 'Or*', 'FU*', 'BY*', 'RS*',
'Pu*', 'RR*', 'Ce*', 'dS*', 'RV*', 'WV*', 'bC*', 'cC*', 'gD*', 'LP*', 'Mi*', 'LP?',
'SN*', 'su*', 'G', 'PoG', 'GiC', 'BiC', 'GiG', 'GiP', 'HzG', 'ALS', 'LyA',
'DLA', 'mAL', 'LLS', 'BAL', 'rG', 'H2G', 'LSB', 'AG?', 'Q?', 'Bz?', 'BL?',
'EmG', 'SBG', 'bCG', 'LeI', 'LeG', 'LeQ', 'AGN', 'LIN', 'SyG', 'Sy1', 'Sy2',
'Bla', 'BLL', 'QSO'] + ['HII',  'No*', 'MoC', 'Cld', 'HH', 'Ae*']

SAVE=True # save as you go
ARCHIVE_UPDATE_FILE = 'consumed_files.txt' 


CANDIDS_JSON_PATH = '/epyc/users/ykwang/scripts/candids_test.json'
oid_csv_path = '/epyc/users/ykwang/scripts/object_data.csv'
LC_FIELDS = ["jd", "fid", "magpsf", "sigmapsf", "diffmaglim", "isdiffpos", "magnr", "sigmagnr", "field", "rcid"]
OID_FIELDS = ["ra", "dec", "ssdistnr", "elong", "objectidps1", "distpsnr1", "sgmag1", "srmag1", "simag1"]
LC_UPDATE_FILE='lc_full_status.txt'