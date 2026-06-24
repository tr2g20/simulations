import numpy as np
import h5py
from pathlib import Path


def save_p_dists(file_path: Path, p_list: list, weights_list: list, init_temp, n_atoms: int, basis: np.ndarray, init_state: np.ndarray, cycles: int, pumping_route: str, date: str):
    """
    Saves lists of momentum distributions and weights, as well as simulation metadata to a HDF5 file.

    Automatically creates parent directories if they do not exist. Prevents 
    overwriting by appending an incremental counter to the filename if a file 
    with the target name is already present.

    Args:
        file_path (Path): Destination path for the HDF5 file.
        p_list (list): List of 1D arrays representing momentum distributions.
        weights_list (list): List of 1D arrays recording the probability fractions corresponding to p_list.
        init_temp (float): Initial temperature in Kelvin.
        n_atoms (int): Number of atoms in the ensemble.
        basis (np.ndarray): The 1D array of momentum basis states (in integers of hbar*k_eff).
        init_state (np.ndarray): The initial momentum state vector of a single atom (elements are np.complex128).
        cycles (int): Number of simulation cycles executed.
        pumping_route (str): Identifier for the pumping route used.
        date (str): Creation date or timestamp for the dataset.
    """

    file_path.parent.mkdir(parents=True, exist_ok=True) # creates directories in file path if they dont exist

    counter = 0
    current_path = file_path

    while True:
        try:

            with h5py.File(current_path, 'w-') as f:
                f.attrs['initial_temp'] = init_temp
                f.attrs['n_atoms'] = n_atoms
                f.attrs['date_created'] = date
                f.attrs['basis'] = basis
                f.attrs['initial_state'] = init_state
                f.attrs['cycles'] = cycles
                f.attrs['pumping_route'] = pumping_route

                group1 = f.create_group('p_list')
                for i, arr in enumerate(p_list):
                    group1.create_dataset(f"{i:04d}", data=arr) # 4 digit indentifier for each array in list
                group2 = f.create_group('weights_list')
                for i, arr in enumerate(weights_list):
                    group2.create_dataset(f"{i:04d}", data=arr)
            break

        except FileExistsError: # increments counter and adds to end of file name if file exists with same name
            counter += 1
            new_name = f"{file_path.stem}_{counter}{file_path.suffix}"
            current_path = file_path.with_name(new_name)

def load_p_dists(file_path: Path) -> dict:
    """
    Loads lists of momentum distributions and weights, as well as simulation metadata from a HDF5 file.

    Args:
        file_path (Path): Path to the HDF5 file to be read.

    Returns:
        dict: A dictionary containing the extracted data with the following keys:
            - 'p_list' (list): List of 1D arrays representing momentum distributions.
            - 'weights_list' (list): List of 1D arrays recording the probability fractions corresponding to p_list.
            - 'init_temp' (float): Initial temperature in Kelvin.
            - 'n_atoms' (int): Number of atoms in the ensemble.
            - 'date_created' (str): Creation date or timestamp for the dataset.
            - 'basis' (np.ndarray): The 1D array of momentum basis states (in integers of hbar*k_eff).
            - 'initial_state' (np.ndarray): The initial momentum state vector of a single atom (elements are np.complex128).
            - 'cycles' (int): Number of simulation cycles executed.
            - 'pumping_route' (str): Identifier for the pumping route used.
    """

    with h5py.File(file_path, 'r') as f:
        init_temp = f.attrs['initial_temp']
        n_atoms = f.attrs['n_atoms']
        date_created = f.attrs['date_created']
        basis = f.attrs['basis']
        init_state = f.attrs['initial_state']
        cycles = f.attrs['cycles']
        pumping_route = f.attrs['pumping_route']
        
        # sorted ensures "0000" comes before "0001"
        p_list = [f['p_list'][key][()] for key in sorted(f['p_list'].keys())]
        weights_list = [f['weights_list'][key][()] for key in sorted(f['weights_list'].keys())]
        
        
    return {
        'p_list': p_list,
        'weights_list': weights_list,
        'init_temp': init_temp,
        'n_atoms': n_atoms,
        'date_created': date_created,
        'basis': basis,
        'initial_state': init_state,
        'cycles': cycles,
        'pumping_route': pumping_route,
    }