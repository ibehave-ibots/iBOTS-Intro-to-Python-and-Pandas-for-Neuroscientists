from functools import lru_cache
from pathlib import Path
from typing import Any
from warnings import warn
import numpy as np
import pandas as pd
from tqdm import tqdm
from xarray import DataArray, Dataset, Coordinates


@lru_cache
def get_brain_group_dict() -> dict[str, str]:
    brain_groups = {}
    for area in ["VISa", "VISam", "VISl", "VISp", "VISpm", "VISrl"]:
        brain_groups[area] = 'visual cortex'
    for area in ["CL", "LD", "LGd", "LH", "LP", "MD", "MG", "PO", "POL", "PT", "RT", "SPF", "TH", "VAL", "VPL", "VPM"]:
        brain_groups[area] = 'thalamus'
    for area in ["CA", "CA1", "CA2", "CA3", "DG", "SUB", "POST"]:
        brain_groups[area] = 'hippocampus'
    for area in ["ACA", "AUD", "COA", "DP", "ILA", "MOp", "MOs", "OLF", "ORB", "ORBm", "PIR", "PL", "SSp", "SSs", "RSP"," TT"]:
        brain_groups[area] = 'non-visual cortex'
    for area in ["APN", "IC", "MB", "MRN", "NB", "PAG", "RN", "SCs", "SCm", "SCig", "SCsg", "ZI"]:
        brain_groups[area] = 'midbrain'
    for area in ["ACB", "CP", "GPe", "LS", "LSc", "LSr", "MS", "OT", "SNr", "SI"]:
        brain_groups[area] = 'basal ganglia'
    for area in ["BLA", "BMA", "EP", "EPd", "MEA"]:
        brain_groups[area] = 'cortical subplate'
    return brain_groups



def steinmetz_to_xarray(dd_part: dict[str, Any], dd_wav: dict[str, Any], dd_lfp: dict[str, Any], dd_st: dict[str, Any]) -> Dataset:
    assert list(dd_part['ccf_axes']) == ['ap', 'dv', 'lr']

    spike_events_df = steinmetz_to_spiketimes_dataframe(dd_st=dd_st)

    dset = Dataset(
        dict(
            # Stimulus Data
            contrast_left = DataArray(
                data=(np.concatenate(
                    (dd_part['contrast_left'], dd_part['contrast_left_passive']),
                ) * 100).astype(np.int8),
                dims=('trial',)
            ),
            contrast_right = DataArray(
                data=(np.concatenate(
                    (dd_part['contrast_right'], dd_part['contrast_right_passive']),
                ) * 100).astype(np.int8),
                dims=('trial',)
            ),
            gocue = DataArray(
                data=np.concatenate((dd_part['gocue'].squeeze(), [np.nan] * dd_part['licks_passive'].shape[1])), 
                dims=('trial',),
            ),
            stim_onset = DataArray(
                data=np.repeat([dd_part['stim_onset']], repeats=dd_part['active_trials'].shape[0]),
                dims=('trial'),
            ),
            feedback_type = DataArray(
                data=np.concatenate((dd_part['feedback_type'].squeeze(), [np.nan] * dd_part['licks_passive'].shape[1])), 
                dims=('trial',),
            ),
            feedback_time = DataArray(
                data=np.concatenate((dd_part['feedback_time'].squeeze(), [np.nan] * dd_part['licks_passive'].shape[1])), 
                dims=('trial',),
            ),
            response_type = DataArray(
                data=np.concatenate((dd_part['response'].squeeze(), [np.nan] * dd_part['licks_passive'].shape[1])), 
                dims=('trial',),
            ),
            response_time = DataArray(
                data=np.concatenate((dd_part['response_time'].squeeze(), [np.nan] * dd_part['licks_passive'].shape[1])), 
                dims=('trial',),
            ),
            reaction_type = DataArray(
                data=np.concatenate((dd_part['reaction_time'][:, 1], [np.nan] * dd_part['licks_passive'].shape[1])), 
                dims=('trial',),
            ),
            reaction_time = DataArray(
                data=np.concatenate((dd_part['reaction_time'][:, 0], [np.nan] * dd_part['licks_passive'].shape[1])), 
                dims=('trial',),
            ),
            prev_reward = DataArray(
                data=np.concatenate((dd_part['prev_reward'].squeeze(), [np.nan] * dd_part['licks_passive'].shape[1])), 
                dims=('trial',),
            ),
            active_trials = DataArray(data=dd_part['active_trials'], dims=('trial',)),

            # Wheel data
            wheel = DataArray(
                data=np.concatenate(
                    (dd_part['wheel'].squeeze(), dd_part['wheel_passive'].squeeze()), 
                    axis=0,
                ).astype(np.int8),
                dims=('trial', 'time')
            ),

            # Licks data
            licks = DataArray(
                data=np.concatenate(
                    (dd_part['licks'].squeeze(), dd_part['licks_passive'].squeeze()),
                    axis=0,
                ).astype(np.int8),
                dims=('trial', 'time'),
            ),

            # Pupil data
            pupil_x = DataArray(
                data=np.concatenate(
                    (dd_part['pupil'][1, :, :], dd_part['pupil_passive'][1, :, :]),
                    axis=0,
                ),
                dims=('trial', 'time')
            ),
            pupil_y = DataArray(
                data=np.concatenate(
                    (dd_part['pupil'][2, :, :], dd_part['pupil_passive'][2, :, :]),
                    axis=0,
                ),
                dims=('trial', 'time')
            ),
            pupil_area = DataArray(
                data=np.concatenate(
                    (dd_part['pupil'][0, :, :], dd_part['pupil_passive'][0, :, :]),
                    axis=0,
                ),
                dims=('trial', 'time')
            ),

            # Face data
            face = DataArray(
                data=np.concatenate(
                    (dd_part['face'].squeeze(), dd_part['face_passive'].squeeze()),
                    axis=0,
                ),
                dims=('trial', 'time'),
            ),

            # Spike data
            spike_rate = DataArray(
                data=np.concatenate(
                    (dd_part['spks'], dd_part['spks_passive']),
                    axis=1,
                ).astype(np.int8), 
                dims=('cell', 'trial', 'time')
            ),

            trough_to_peak = DataArray(data=dd_part['trough_to_peak'].astype(np.int8), dims=('cell',)),
            ccf_ap = DataArray(data=dd_part['ccf'][:, 0], dims=('cell',)),
            ccf_dv = DataArray(data=dd_part['ccf'][:, 1], dims=('cell',)),
            ccf_lr = DataArray(data=dd_part['ccf'][:, 2], dims=('cell',)),
            brain_area = DataArray(data=dd_part['brain_area'], dims=('cell',)),
            
            brain_groups = DataArray(
                data=[get_brain_group_dict().get(area, area) for area in dd_part['brain_area']],
                dims=('cell',)
            ),

            # Waveform data
            waveform_w = DataArray(
                data=dd_wav['waveform_w'],
                dims=('cell', 'sample', 'waveform_component'),
            ),
            waveform_u = DataArray(
                data=dd_wav['waveform_u'],
                dims=('cell', 'waveform_component', 'probe'),
            ),

            # LFP data
            lfp = DataArray(
                data=np.concatenate((dd_lfp['lfp'], dd_lfp['lfp_passive']), axis=1),
                dims=('brain_area_lfp', 'trial', 'time'),
            ),

            # Raw Spike Events Data
            
            # df = pd.DataFrame(rows, columns=['Cell', 'Trial', 'SpikeTime'])
            spike_time = DataArray(
                data=spike_events_df['SpikeTime'].values,
                dims=('spike_id',)
            ),
            spike_cell = DataArray(
                data=spike_events_df['Cell'].values,
                dims=('spike_id',)
            ),
            spike_trial = DataArray(
                data=spike_events_df['Trial'].values,
                dims=('spike_id',)
            ),
            
        ),
        coords=Coordinates({
            'trial': np.arange(1, dd_part['active_trials'].shape[0] + 1),
            'time': (np.arange(1, dd_part['wheel'].shape[-1] + 1) * dd_part['bin_size']),
            'cell': np.arange(1, dd_part['spks'].shape[0] + 1),
            'waveform_component': np.arange(1, dd_wav['waveform_w'].shape[2] + 1),
            'probe': np.arange(1, dd_wav['waveform_u'].shape[2] + 1),
            'brain_area_lfp': dd_lfp['brain_area_lfp'],
            'spike_id': np.arange(1, len(spike_events_df) + 1)
        }),
        attrs={
            'session_date': dd_part['date_exp'],
            'mouse': dd_part['mouse_name'],
            'stim_onset': dd_part['stim_onset'],
            'bin_size': dd_part['bin_size'],
        }
    )
    # .expand_dims({
    #     'mouse': [dd_part['mouse_name']],
    #     'session_date': [dd_part['date_exp']],
    # })
    return dset


def steinmetz_to_spiketimes_dataframe(dd_st: dict[str, Any]) -> pd.DataFrame:
    spike_times = np.concatenate((dd_st['ss'], dd_st['ss_passive']), axis=1)
    
    rows = []
    for neuron_id, neuron_data in enumerate(spike_times, start=1):
        for trial_id, trial_data in enumerate(neuron_data, start=1):
            for spike_time in trial_data:
                rows.append([neuron_id, trial_id, spike_time])

    df = pd.DataFrame(rows, columns=['Cell', 'Trial', 'SpikeTime'])
    df = df.astype({'Trial': np.uint32, 'Cell': np.uint32})
    return df



if __name__ == '__main__':

    base_path = Path('data/processed')
    base_path.mkdir(parents=True, exist_ok=True)

    print('Reading Spike Times File...', end='', flush=True)
    dat_st = iter(np.load('data/raw/lfp/steinmetz_st.npz', allow_pickle=True)['dat'])
    print(f'..done.', flush=True)

    print('Reading Spike Waveform Data...', end='', flush=True)
    dat_wav = iter(np.load('data/raw/lfp/steinmetz_wav.npz', allow_pickle=True)['dat'])
    print(f'..done.', flush=True)

    print('Reading LFP Data...', end='', flush=True)
    dat_lfp = iter(np.load('data/raw/lfp/steinmetz_lfp.npz', allow_pickle=True)['dat'])
    print(f'..done.', flush=True)

    paths = [Path(f'data/raw/neuropixels/steinmetz_part{i}.npz') for i in [0, 1, 2]]
    for path in tqdm(paths, desc="Reading Raw NPZ Files"):
        dat = np.load(path, allow_pickle=True)['dat']

        for dd, dd_st, dd_wav, dd_lfp in tqdm(list(zip(dat, dat_st, dat_wav, dat_lfp)), desc=f"Writing Processed NetCDF Files from {path.name}"):

            # Verify that the sessions in different files match, using cell counts
            if dd_wav['waveform_w'].shape[0] != dd['cellid_orig'].sum():
                raise IOError(f"Problem at {dd['date_exp'], dd['mouse_name']}.  Reason: has a different number of cells in the partx.npx and extra.npx data files")


            # Make xarray Dataset for most data, save to compressed NetCDF file.
            dset = steinmetz_to_xarray(dd_part=dd, dd_wav=dd_wav, dd_lfp=dd_lfp, dd_st=dd_st)            
            # settings = {'zlib': True, 'complevel': 5}  # Compression settings for each variable. Slower to write, but shrunk data to 6% the original size!
            # encodings = {var: settings for var in dset.data_vars if not 'U' in str(dset[var].dtype)}
            
            session_path = base_path / f'steinmetz_{dd["date_exp"]}_{dd["mouse_name"]}.nc'
            session_path.parent.mkdir(parents=True, exist_ok=True)
            dset.to_netcdf(
                path=session_path,
                format="NETCDF4",
                engine="netcdf4",
                # encoding=encodings,   
            )