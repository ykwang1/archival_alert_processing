import pandas as pd
import numpy as np

from os import listdir
from os.path import isfile, join
DATA_PATH = '/epyc/users/ykwang/data/ac_lc_full/'
data_files = [f for f in listdir(DATA_PATH) if isfile(join(DATA_PATH, f))]
data_files = pd.Series(data_files).sort_values().values

df = pd.DataFrame([])
for ii,chunk in enumerate(np.array_split(data_files, 10)):
    df = pd.DataFrame([])
    ts = chunk[0].split('_')[0]
    for f in chunk:
        temp = pd.read_csv(DATA_PATH + f)
        print(f)
        df = pd.concat([df, temp])
        df.drop_duplicates(inplace=True)
    print(f'saving chunk {ii}')
    df.to_csv(f'{DATA_PATH}../{ts}.csv', index=False)
