from amplitude_cut import *
from datetime import date

ARCHIVAL_DIR = '/epyc/data/ztf/alerts/'
SAVE_DIR = '~/notebooks/data/ac_archival/'

today = date.today()
ts = today.strftime("%Y%m%d")
ts = '20220627'
print(f'starting {ts}...')
for program in ['public', 'partnership']:
    df = consume_one_night(str(ts).strip())
    df.to_csv(f'{SAVE_DIR}{ts}_{program}.csv', index=False)
    with open(update_file, 'a') as g:
        g.write(f'{ts} written\n')