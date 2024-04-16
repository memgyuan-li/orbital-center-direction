'''
Author: memgyuan-li
Date: 2024-04-14 19:37:46
LastEditTime: 2024-04-16 21:54:03
Description: 
Bug fixed of d-band-center.py
Atoms without spin
Calculate the determined track direction
ref 'https://github.com/hitarth64/d-band-center'
'''


import numpy as np
import pandas as pd
from scipy.integrate import simps
import datetime
import time


def pdos_column_names(lmax, ispin):
    if lmax == 2:
        names = ['s', 'p_y', 'p_z', 'p_x', 'd_xy', 'd_yz', 'd_z2-r2', 'd_xz', 'd_x2-y2']
        # names = [ 's', 'py', 'pz', 'px', 'dxy', 'dyz', 'dz2', 'dxz', 'dx2-y2' ]
    elif lmax == 3:
        names = ['s', 'p_y', 'p_z', 'p_x', 'd_xy', 'd_yz', 'd_z2-r2', 'd_xz', 'd_x2-y2',
                 'f_y(3x2-y2)', 'f_xyz', 'f_yz2', 'f_z3', 'f_xz2', 'f_z(x2-y2)', 'f_x(x2-3y2)']
        # names = [ 's', 'py', 'pz', 'px', 'dxy', 'dyz', 'dz2', 'dxz', 'dx2-y2','f-3',
        #         'f-2', 'f-1', 'f0', 'f1', 'f2', 'f3']
    else:
        raise ValueError('lmax value not supported')

    if ispin == 2:
        all_names = []
        for n in names:
            all_names.extend(['{}_up'.format(n), '{}_down'.format(n)])
    else:
        all_names = names
    all_names.insert(0, 'energy')
    return all_names


class Doscar:
    '''
    Contains all the data in a VASP DOSCAR file, and methods for manipulating this.
    '''

    number_of_header_lines = 6

    def __init__(self, filename, ispin=1, lmax=2, lorbit=11, spin_orbit_coupling=False, read_pdos=True, species=None):
        '''
        Create a Doscar object from a VASP DOSCAR file.
        Args:
            filename (str): Filename of the VASP DOSCAR file to read.
            ispin (optional:int): ISPIN flag. 
                Set to 1 for non-spin-polarised or 2 for spin-polarised calculations.
                Default = 2.
            lmax (optional:int): Maximum l angular momentum. (d=2, f=3). Default = 2.
            lorbit (optional:int): The VASP LORBIT flag. (Default=11).
            spin_orbit_coupling (optional:bool): Spin-orbit coupling (Default=False).
            read_pdos (optional:bool): Set to True to read the atom-projected density of states (Default=True).
            species (optional:list(str)): List of atomic species strings, e.g. [ 'Fe', 'Fe', 'O', 'O', 'O' ].
                Default=None.
        '''
        self.filename = filename
        self.ispin = ispin
        self.lmax = lmax
        self.spin_orbit_coupling = spin_orbit_coupling
        if self.spin_orbit_coupling:
            raise NotImplementedError('Spin-orbit coupling is not yet implemented')
        self.lorbit = lorbit
        self.pdos = None
        self.species = species
        self.read_header()
        self.read_total_dos()
        if read_pdos:
            try:
                self.read_projected_dos()
            except:
                raise
        # if species is set, should check that this is consistent with the number of entries in the
        # projected_dos dataset

    @property
    def number_of_channels(self):
        if self.lorbit == 11:
            return {2: 9, 3: 16}[self.lmax]
        raise NotImplementedError

    def read_header(self):
        self.header = []
        with open(self.filename, 'r') as file_in:
            for i in range(Doscar.number_of_header_lines):
                self.header.append(file_in.readline())
        self.process_header()

    def process_header(self):
        self.number_of_atoms = int(self.header[0].split()[0])
        self.number_of_data_points = int(self.header[5].split()[2])
        self.efermi = float(self.header[5].split()[3])

    def read_total_dos(self):  # assumes spin_polarised
        start_to_read = Doscar.number_of_header_lines
        df = pd.read_csv(self.filename,
                         skiprows=start_to_read,
                         nrows=self.number_of_data_points,
                         delim_whitespace=True,
                         names=['energy', 'up', 'down', 'int_up', 'int_down'],
                         index_col=False)
        self.energy = df.energy.values
        df.drop('energy', axis=1)
        self.tdos = df

    def read_atomic_dos_as_df(self, atom_number):  # currently assume spin-polarised, no-SO-coupling, no f-states
        assert atom_number > 0 & atom_number <= self.number_of_atoms
        start_to_read = Doscar.number_of_header_lines + atom_number * (self.number_of_data_points + 1)
        df = pd.read_csv(self.filename,
                         skiprows=start_to_read,
                         nrows=self.number_of_data_points,
                         delim_whitespace=True,
                         names=pdos_column_names(lmax=self.lmax, ispin=self.ispin),
                         index_col=False)
        return df.drop('energy', axis=1)

    def read_projected_dos(self):
        """
        Read the projected density of states data into """
        pdos_list = []
        for i in range(self.number_of_atoms):
            df = self.read_atomic_dos_as_df(i + 1)
            pdos_list.append(df)
        # self.pdos  =   pdos_list
        self.pdos = np.vstack([np.array(df) for df in pdos_list]).reshape(
            self.number_of_atoms, self.number_of_data_points, self.number_of_channels, self.ispin)

    def pdos_select(self, atoms=None, spin=None, l=None, m=None):
        """
        Returns a subset of the projected density of states array.
        """
        valid_m_values = {'s': [],
                          'p': ['x', 'y', 'z'],
                          'd': ['xy', 'yz', 'z2-r2', 'xz', 'x2-y2'],
                          'f': ['y(3x2-y2)', 'xyz', 'yz2', 'z3', 'xz2', 'z(x2-y2)', 'x(x2-3y2)']}
        if not atoms:
            atom_idx = list(range(self.number_of_atoms))
        else:
            atom_idx = [i-1 for i in atoms]
        to_return = self.pdos[atom_idx, :, :, :]
        if not spin:
            spin_idx = list(range(self.ispin))
        elif spin == 'up':
            spin_idx = [0]
        elif spin == 'down':
            spin_idx = [1]
        elif spin == 'both':
            spin_idx = [0, 1]
        else:
            raise ValueError
        to_return = to_return[:, :, :, spin_idx]

        if not l:
            channel_idx = list(range(self.number_of_channels))
        elif l == 's':
            channel_idx = [0]
        elif l == 'p':
            if not m:
                channel_idx = [1, 2, 3]
            else:
                channel_idx = [i + 1 for i, v in enumerate(valid_m_values['p']) if v in m]
        elif l == 'd':
            if not m:
                channel_idx = [4, 5, 6, 7, 8]
            else:
                channel_idx = [i + 4 for i, v in enumerate(valid_m_values['d']) if v in m]
        elif l == 'f':
            if not m:
                channel_idx = [9, 10, 11, 12, 13, 14, 15]
            else:
                channel_idx = [i + 9 for i, v in enumerate(valid_m_values['f']) if v in m]
        else:
            raise ValueError
        return to_return[:, :, channel_idx, :]

    def pdos_sum(self, atoms=None, spin=None, l=None, m=None):
        return np.sum(self.pdos_select(atoms=atoms, spin=spin, l=l, m=m), axis=(0, 2, 3))


# calculation of d-band center
# Open doscar
dosfile = 'DOSCAR'
doscar = Doscar(dosfile, ispin=1, lmax=3, lorbit=11)  # calculation setting
atoms = list(range(5,6))  # calculated atom ordinal

print(*atoms)
# Set atoms for integration
orb = 's'
orip = ['x', 'y', 'z']
orid = ['xy', 'yz', 'z2-r2', 'xz', 'x2-y2']
nospin=doscar.pdos_sum(atoms,l=orb)
efermi = doscar.efermi - doscar.efermi
energies = doscar.energy - doscar.efermi
emin, emax = energies[0], energies[-1]
erange = (efermi + 0, efermi + 2)  # integral energy range
emask = ((energies >= erange[0]) & (energies <= erange[-1]))
x = energies[emask]
y3 = nospin[emask]
i_nospin=simps(y3, x)
dbc_nospin = simps(y3 * x, x) / simps(y3, x)
dbc = []
dbc.append(dbc_nospin)
with open('bandrho.txt', 'a') as f:
    print(i_nospin,dbc_nospin, file=f)
f.close()

# for index in range(len(orip)):
#     # up = doscar.pdos_sum(atoms, spin='up', l=orb, m=orip[index])
#     # down = doscar.pdos_sum(atoms, spin='down', l=orb, m=orip[index])
#     # up = doscar.pdos_sum(atoms, spin='up', l=orb )
#     # down = doscar.pdos_sum(atoms, spin='down', l=orb )
#     # Set intergrating range
#     efermi = doscar.efermi - doscar.efermi
#     energies = doscar.energy - doscar.efermi
#     emin, emax = energies[0], energies[-1]
#     # print(emin,emax)
#     erange = (efermi + 0, efermi + 2)  # integral energy range
#     # print(erange[0])
#     emask = ((energies >= erange[0]) & (energies <= erange[-1]))
#     #print(emask,up.shape)
#     #print(up)
#     x = energies[emask]
#     y1 = up[emask]
#     y2 = down[emask]
#     i_up=simps(y1, x)
#     i_down=simps(y2, x)
#     dbc_up = simps(y1 * x, x) / simps(y1, x)
#     dbc_down = simps(y2 * x, x) / simps(y2, x)
#     dbc_hyb = simps((y2 - y1) * x, x) / simps(y2 - y1, x)
#     dbc = []
#     dbc.append(dbc_up)
#     dbc.append(dbc_down)
#     with open('bandrho.txt', 'a') as f:
#         print(i_up,i_down,dbc_up, dbc_down, file=f)
#     f.close()
