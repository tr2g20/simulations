import numpy as np
import h5py
from pathlib import Path

def save_p_dists(file_path: Path, p_list: list, weights_list: list, init_temp, n_atoms: int, date: str, rng_state = None):
    
    with h5py.File(file_path, 'w') as f:
        f.attrs['initial_temp'] = init_temp
        f.attrs['n_atoms'] = n_atoms
        f.attrs['date_created'] = date

        group1 = f.create_group('p_list')
        for i, arr in enumerate(p_list):
            group1.create_dataset(f"{i:04d}", data=arr) # 4 digit indentifier for each array in list
        group2 = f.create_group('weights_list')
        for i, arr in enumerate(weights_list):
            group2.create_dataset(f"{i:04d}", data=arr)
        if rng_state is not None:
            f.create_dataset('rng_state', data=rng_state)

def load_p_dists(file_path: Path) -> dict:

    with h5py.File(file_path, 'r') as f:
        init_temp = f.attrs['initial_temp']
        n_atoms = f.attrs['n_atoms']
        date_created = f.attrs['date_created']
        
        # sorted ensures "0000" comes before "0001"
        p_list = [f['p_list'][key][()] for key in sorted(f['p_list'].keys())]
        weights_list = [f['weights_list'][key][()] for key in sorted(f['weights_list'].keys())]
        
        rng_state = f['rng_state'][()] if 'rng_state' in f else None
        
    return {
        'p_list': p_list,
        'weights_list': weights_list,
        'rng_state': rng_state,
        'init_temp': init_temp,
        'n_atoms': n_atoms,
        'date_created': date_created,
    }