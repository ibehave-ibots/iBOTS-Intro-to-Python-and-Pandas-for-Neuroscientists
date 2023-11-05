# %%
# !pip install netCDF4 pyarrow

# %%
from glob import glob
from hashlib import md5
from pathlib import Path

import pandas as pd
import xarray as xr

# %%
paths = glob('data/processed/*.nc')
Path('data/final').mkdir(parents=True, exist_ok=True)

# %%
dfs = []
for path in paths:
    dset = xr.load_dataset(path)
    df = dset[['active_trials', 'contrast_left', 'contrast_right', 'stim_onset', 'gocue', 'response_type', 'response_time', 'feedback_time', 'feedback_type', 'reaction_time', 'reaction_type']].to_dataframe().reset_index()
    # df = df[df['active_trials']].drop(columns=['active_trials'])
    df = df.rename(columns={'gocue': 'gocue_time'})
    df = df.assign(
        mouse=dset.attrs['mouse'], 
        session_date=dset.attrs['session_date'], 
        session_id=str(md5((dset.attrs['mouse'] + dset.attrs['session_date']).encode()).hexdigest())[:6],
    )
    dfs.append(df)
df = pd.concat(dfs, ignore_index=True)
df.head()


# %%
df_winter2016 = df[df.session_date.isin(
    ['2016-12-14', '2016-12-17', '2016-12-18', 
     '2017-01-07', '2017-01-08', '2017-01-09', '2017-01-10', '2017-01-11', '2017-01-12',
    ]
)]
df_winter2016.to_csv('data/final/steinmetz_winter2016.csv', index=False)

# %%
df_summer2017 = df[df.session_date.isin(
    ['2017-05-15', '2017-05-16', '2017-05-18',
    '2017-06-15', '2017-06-16', '2017-06-17', '2017-06-18',]
)]
df_summer2017.to_csv('data/final/steinmetz_summer2017.csv', index=False)
# df_summer2017.head()

# %%
df_winter2017 = df[df.session_date.isin(
    ['2017-10-11', '2017-10-29', '2017-10-30', '2017-10-31',
     '2017-11-01', '2017-11-02', '2017-11-04', '2017-11-05',
     '2017-12-05', '2017-12-06', '2017-12-07', '2017-12-08',   '2017-12-09', '2017-12-10', '2017-12-11'
    ]
)]
df_winter2017.to_csv('data/final/steinmetz_winter2017.csv', index=False)

# %%
df.to_parquet('data/final/steinmetz_all.parquet')
df.to_csv('data/final/steinmetz_all.csv', index=False)

# %%
df.session_date.unique()

# %%



