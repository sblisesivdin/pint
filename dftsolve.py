#!/usr/bin/env python

'''
dftsolve.py: High-level Interaction Script for GPAW
More information: $ dftsolve.py -h
'''

Description = f'''
 Usage:
 $ mpirun -np <corenumbers> dftsolve.py <args>
'''

import getopt, sys, os, time, shutil
import textwrap
import requests
import pickle
from argparse import ArgumentParser, HelpFormatter
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from ase import *
from ase.spacegroup import get_spacegroup
from ase.dft.kpoints import get_special_points
from ase.parallel import paropen, world, parprint, broadcast
from gpaw import GPAW, PW, Davidson, FermiDirac, MixerSum, MixerDif, Mixer
from ase.optimize import QuasiNewton
from ase.io import read, write
from ase.calculators.singlepoint import SinglePointCalculator
from ase.eos import calculate_eos
from ase.units import Bohr, GPa, kJ
import matplotlib.pyplot as plt
from ase.constraints import FixSymmetry
from ase.filters import FrechetCellFilter
from ase.io.cif import write_cif
from pathlib import Path
from gpaw.response.df import DielectricFunction
from gpaw.response.bse import BSE
from gpaw.dos import DOSCalculator
from gpaw.utilities.dos import raw_orbital_LDOS
import numpy as np
from numpy import genfromtxt
from elastic import get_elastic_tensor, get_elementary_deformations
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
import phonopy
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

@dataclass
class DFTConfig:
    """Configuration dataclass to hold all DFT calculation parameters.
    
    This dataclass replaces the previous global variable approach, providing:
    - Type hints for better code clarity
    - Default values in one centralized location
    - Easier testing and parameter management
    - Reduced global namespace pollution
    
    All parameters that were previously global variables are now encapsulated
    in this dataclass. The dftsolve class receives a DFTConfig instance and
    exposes its attributes for backward compatibility.
    """
    # Mode and calculation flags
    Mode: str = 'PW'
    Ground_calc: bool = False
    Geo_optim: bool = False
    Elastic_calc: bool = False
    DOS_calc: bool = False
    Band_calc: bool = False
    Density_calc: bool = False
    Phonon_calc: bool = False
    Optical_calc: bool = False
    
    # Geometry optimization parameters
    Optimizer: str = 'QuasiNewton'
    Max_F_tolerance: float = 0.05
    Max_step: float = 0.1
    Alpha: float = 60.0
    Damping: float = 1.0
    Fix_symmetry: bool = False
    Relax_cell: List[bool] = field(default_factory=lambda: [False, False, False, False, False, False])
    Hydrostatic_pressure: float = 0.0
    
    # Ground state parameters
    Cut_off_energy: float = 340
    Ground_kpts_density: Optional[float] = None
    Ground_kpts_x: int = 5
    Ground_kpts_y: int = 5
    Ground_kpts_z: int = 5
    Ground_gpts_density: Optional[float] = None
    Ground_gpts_x: int = 8
    Ground_gpts_y: int = 8
    Ground_gpts_z: int = 8
    Setup_params: Dict = field(default_factory=dict)
    XC_calc: str = 'LDA'
    Ground_convergence: Dict = field(default_factory=dict)
    Occupation: Dict = field(default_factory=lambda: {'name': 'fermi-dirac', 'width': 0.05})
    Mixer_type: Any = None
    Spin_calc: bool = False
    Magmom_per_atom: float = 1.0
    Magmom_single_atom: Optional[List] = None
    Total_charge: float = 0.0
    
    # DOS parameters
    DOS_npoints: int = 501
    DOS_width: float = 0.1
    DOS_convergence: Dict = field(default_factory=dict)
    
    # Band structure parameters
    Gamma: bool = True
    Band_path: str = 'LGL'
    Band_npoints: int = 61
    Energy_max: float = 5
    Energy_min: float = -5
    Band_convergence: Dict = field(default_factory=lambda: {'bands': 8})
    
    # Electron density parameters
    Refine_grid: int = 4
    
    # Phonon parameters
    Phonon_PW_cutoff: float = 400
    Phonon_kpts_x: int = 3
    Phonon_kpts_y: int = 3
    Phonon_kpts_z: int = 3
    Phonon_supercell: Any = None
    Phonon_displacement: float = 1e-3
    Phonon_path: str = 'LGL'
    Phonon_npoints: int = 61
    Phonon_acoustic_sum_rule: bool = True
    
    # Optical parameters
    Opt_calc_type: str = 'BSE'
    Opt_shift_en: float = 0.0
    Opt_BSE_valence: Any = None
    Opt_BSE_conduction: Any = None
    Opt_BSE_min_en: float = 0.0
    Opt_BSE_max_en: float = 20.0
    Opt_BSE_num_of_data: int = 1001
    Opt_num_of_bands: int = 8
    Opt_FD_smearing: float = 0.05
    Opt_eta: float = 0.05
    Opt_domega0: float = 0.05
    Opt_omega2: float = 5.0
    Opt_cut_of_energy: float = 100
    Opt_nblocks: Any = None
    
    # General parameters
    MPI_cores: int = 4
    Localisation: str = "en_UK"
    Outdirname: str = ''
    
    # Bulk configuration
    bulk_configuration: Any = None
    
    # Localization labels
    dos_xlabel: Dict = field(default_factory=dict)
    dos_ylabel: Dict = field(default_factory=dict)
    band_ylabel: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize default values that depend on other objects."""
        if self.Mixer_type is None:
            self.Mixer_type = MixerSum(0.1, 3, 50)
        if self.Phonon_supercell is None:
            self.Phonon_supercell = np.diag([2, 2, 2])
        if self.Opt_BSE_valence is None:
            self.Opt_BSE_valence = range(0, 3)
        if self.Opt_BSE_conduction is None:
            self.Opt_BSE_conduction = range(4, 7)
        if self.Opt_nblocks is None:
            self.Opt_nblocks = world.size

class RawFormatter(HelpFormatter):
    """To print Description variable with argparse"""
    def _fill_text(self, text, width, indent):
        return "\n".join([textwrap.fill(line, width) for line in textwrap.indent(textwrap.dedent(text), indent).splitlines()])

def struct_from_file(inputfile, geometryfile):
    """Load variables from parse function and return DFTConfig instance."""
    # Works like from FILE import *
    inputf = __import__(Path(inputfile).stem, globals(), locals(), ['*'])
    
    # Create a config object with loaded parameters
    config_dict = {}
    for k in dir(inputf):
        if not k.startswith('_'):
            config_dict[k] = getattr(inputf, k)
            # Also set as globals for backward compatibility with methods that haven't been refactored yet
            globals()[k] = getattr(inputf, k)
    
    # Create DFTConfig instance
    config = DFTConfig(**{k: v for k, v in config_dict.items() if k in DFTConfig.__dataclass_fields__})
    
    # If there is a CIF input, use it. Otherwise use the bulk configuration provided above.
    if geometryfile is None:
        if config.Outdirname != '':
            struct = config.Outdirname
        else:
            struct = 'results' # All files will get their names from this file
    else:
        struct = Path(geometryfile).stem
        config.bulk_configuration = read(geometryfile, index='-1')
        parprint("Number of atoms imported from CIF file:"+str(config.bulk_configuration.get_global_number_of_atoms()))
        parprint("Spacegroup of CIF file:",get_spacegroup(config.bulk_configuration, symprec=1e-2))
        parprint("Special Points usable for this spacegroup:",get_special_points(config.bulk_configuration.get_cell()))

    # Output directory
    if config.Outdirname != '':
        structpath = os.path.join(os.getcwd(), config.Outdirname)
    else:
        structpath = os.path.join(os.getcwd(), struct)

    if not os.path.isdir(structpath):
        os.makedirs(structpath, exist_ok=True)
    struct = os.path.join(structpath, struct)
    return struct, config

def autoscale_y(ax,margin=0.1):
    """This function rescales the y-axis based on the data that is visible given the current xlim of the axis.
    ax -- a matplotlib axes object
    margin -- the fraction of the total height of the y-data to pad the upper and lower ylims"""

    import numpy as np

    def get_bottom_top(line):
        xd = line.get_xdata()
        yd = line.get_ydata()
        lo,hi = ax.get_xlim()
        y_displayed = yd[((xd>lo) & (xd<hi))]
        h = np.max(y_displayed) - np.min(y_displayed)
        bot = np.min(y_displayed)-margin*h
        top = np.max(y_displayed)+margin*h
        return bot,top

    lines = ax.get_lines()
    bot,top = np.inf, -np.inf

    for line in lines:
        new_bot, new_top = get_bottom_top(line)
        if new_bot < bot: bot = new_bot
        if new_top > top: top = new_top

    ax.set_ylim(bot,top)

class dftsolve:
    """
    The dftsolve class is a high-level interaction script for GPAW calculations.
    It handles various types of calculations such as ground state, structure optimization,
    elastic properties, density of states, band structure, density, and optical properties.
    The class takes input parameters from a DFTConfig instance and performs the calculations
    accordingly.
    """
    def __init__(self, struct: str, config: DFTConfig):
        """Initialize dftsolve with struct path and configuration.
        
        Args:
            struct: Path to structure files
            config: DFTConfig instance containing all calculation parameters
        """
        self.struct = struct
        self.config = config
        
        # For backward compatibility, expose config attributes as instance attributes
        self.Mode = config.Mode
        self.Geo_optim = config.Geo_optim
        self.Elastic_calc = config.Elastic_calc
        self.DOS_calc = config.DOS_calc
        self.Band_calc = config.Band_calc
        self.Density_calc = config.Density_calc
        self.Optical_calc = config.Optical_calc
        self.Optimizer = config.Optimizer
        self.Max_F_tolerance = config.Max_F_tolerance
        self.Max_step = config.Max_step
        self.Alpha = config.Alpha
        self.Damping = config.Damping
        self.Fix_symmetry = config.Fix_symmetry
        self.Relax_cell = config.Relax_cell
        self.Hydrostatic_pressure = config.Hydrostatic_pressure
        self.Cut_off_energy = config.Cut_off_energy
        self.Ground_kpts_density = config.Ground_kpts_density
        self.Ground_kpts_x = config.Ground_kpts_x
        self.Ground_kpts_y = config.Ground_kpts_y
        self.Ground_kpts_z = config.Ground_kpts_z
        self.Ground_gpts_density = config.Ground_gpts_density
        self.Ground_gpts_x = config.Ground_gpts_x
        self.Ground_gpts_y = config.Ground_gpts_y
        self.Ground_gpts_z = config.Ground_gpts_z
        self.Setup_params = config.Setup_params
        self.XC_calc = config.XC_calc
        self.Ground_convergence = config.Ground_convergence
        self.Occupation = config.Occupation
        self.Mixer_type = config.Mixer_type
        self.Spin_calc = config.Spin_calc
        self.Magmom_per_atom = config.Magmom_per_atom
        self.Magmom_single_atom = config.Magmom_single_atom
        self.Total_charge = config.Total_charge
        self.DOS_npoints = config.DOS_npoints
        self.DOS_width = config.DOS_width
        self.DOS_convergence = config.DOS_convergence
        self.Gamma = config.Gamma
        self.Band_path = config.Band_path
        self.Band_npoints = config.Band_npoints
        self.Energy_max = config.Energy_max
        self.Energy_min = config.Energy_min
        self.Band_convergence = config.Band_convergence
        self.Refine_grid = config.Refine_grid
        self.Phonon_PW_cutoff = config.Phonon_PW_cutoff
        self.Phonon_kpts_x = config.Phonon_kpts_x
        self.Phonon_kpts_y = config.Phonon_kpts_y
        self.Phonon_kpts_z = config.Phonon_kpts_z
        self.Phonon_supercell = config.Phonon_supercell
        self.Phonon_displacement = config.Phonon_displacement
        self.Phonon_path = config.Phonon_path
        self.Phonon_npoints = config.Phonon_npoints
        self.Phonon_acoustic_sum_rule = config.Phonon_acoustic_sum_rule
        self.Opt_calc_type = config.Opt_calc_type
        self.Opt_shift_en = config.Opt_shift_en
        self.Opt_BSE_valence = config.Opt_BSE_valence
        self.Opt_BSE_conduction = config.Opt_BSE_conduction
        self.Opt_BSE_min_en = config.Opt_BSE_min_en
        self.Opt_BSE_max_en = config.Opt_BSE_max_en
        self.Opt_BSE_num_of_data = config.Opt_BSE_num_of_data
        self.Opt_num_of_bands = config.Opt_num_of_bands
        self.Opt_FD_smearing = config.Opt_FD_smearing
        self.Opt_eta = config.Opt_eta
        self.Opt_domega0 = config.Opt_domega0
        self.Opt_omega2 = config.Opt_omega2
        self.Opt_cut_of_energy = config.Opt_cut_of_energy
        self.Opt_nblocks = config.Opt_nblocks
        self.MPI_cores = config.MPI_cores
        self.Localisation = config.Localisation
        self.bulk_configuration = config.bulk_configuration
        self.dos_xlabel = config.dos_xlabel
        self.dos_ylabel = config.dos_ylabel
        self.band_ylabel = config.band_ylabel
        
        if not self.dos_xlabel:
            self.dos_xlabel.update({
                'en_UK':'Energy [eV]', 'tr_TR':'Enerji [eV]', 'de_DE':'Energie [eV]', 
                'fr_FR':'Énergie [eV]', 'ru_RU':'Энергия [эВ]', 'zh_CN':'能量 [eV]', 
                'ko_KR':'에너지 [eV]', 'ja_JP':'エネルギー [eV]'
            })
        if not self.dos_ylabel:
            self.dos_ylabel.update({
                'en_UK':'DOS [1/eV]', 'tr_TR':'Durum Yoğunluğu [1/eV]', 'de_DE':'Zustandsdichte [1/eV]',
                'fr_FR':'DOS [1/eV]', 'ru_RU':'Плотность состояний [1/эВ]', 'zh_CN':'态密度 [1/eV]',
                'ko_KR':'상태 밀도 [1/eV]', 'ja_JP':'状態密度 [1/eV]'
            })
        if not self.band_ylabel:
            self.band_ylabel.update({
                'en_UK':'Energy [eV]', 'tr_TR':'Enerji [eV]', 'de_DE':'Energie [eV]',
                'fr_FR':'Énergie [eV]', 'ru_RU':'Энергия [эВ]', 'zh_CN':'能量 [eV]',
                'ko_KR':'에너지 [eV]', 'ja_JP':'エネルギー [eV]'
            })
        

    def structurecalc(self):
        """
        This method calculates and writes the spacegroup and special points of the given structure.
        It reads the bulk configuration from the CIF file and prints the number of atoms, spacegroup,
        and special points usable for the spacegroup to a text file.
        """

        # -------------------------------------------------------------
        # Step 0 - STRUCTURE
        # -------------------------------------------------------------

        with paropen(self.struct+'-0-Result-Spacegroup-and-SpecialPoints.txt', "w") as fd:
            print("Number of atoms imported from CIF file:"+str(self.bulk_configuration.get_global_number_of_atoms()), file=fd)
            print("Spacegroup of CIF file:",get_spacegroup(self.bulk_configuration, symprec=1e-2), file=fd)
            print("Special Points usable for this spacegroup:",get_special_points(self.bulk_configuration.get_cell()), file=fd)

    def groundcalc(self):
        """
        This method performs ground state calculations for the given structure using various settings
        and parameters specified in the configuration file. It handles different XC functionals,
        spin calculations, and geometry optimizations. The results are saved in appropriate files,
        including the final configuration as a CIF file and the ground state results in a GPW file.
        """

        # -------------------------------------------------------------
        # Step 1 - GROUND STATE
        # -------------------------------------------------------------

        # Start ground state timing
        time11 = time.time()
        if self.Mode == 'PW':
            if self.Spin_calc == True:
                if self.Magmom_single_atom is not None:
                    numm = [0.0]*self.bulk_configuration.get_global_number_of_atoms()
                    numm[self.Magmom_single_atom[0]] = self.Magmom_single_atom[1]
                else:
                    numm = [self.Magmom_per_atom]*self.bulk_configuration.get_global_number_of_atoms()
                self.bulk_configuration.set_initial_magnetic_moments(numm)
            if self.config.Ground_calc == True:
                # PW Ground State Calculations
                parprint("Starting PW ground state calculation...")
                if True in self.Relax_cell:
                    if self.XC_calc in ['GLLBSC', 'GLLBSCM', 'HSE06', 'HSE03','B3LYP', 'PBE0','EXX']:
                        parprint("\033[91mERROR:\033[0m Structure optimization LBFGS can not be used with "+self.XC_calc+" xc.")
                        parprint("Do manual structure optimization, or do with PBE, then use its final CIF as input.")
                        parprint("Quiting...")
                        quit()
                if self.XC_calc in ['HSE06', 'HSE03','B3LYP', 'PBE0','EXX']:
                    parprint('Starting Hybrid XC calculations...')
                    if self.Ground_kpts_density is not None:
                        calc = GPAW(mode=PW(ecut=self.Cut_off_energy, force_complex_dtype=True), xc={'name': self.XC_calc, 'backend': 'pw'}, nbands='200%',
                                parallel={'band': 1, 'kpt': 1}, eigensolver=Davidson(niter=1), mixer=self.Mixer_type, charge=self.Total_charge,
                                spinpol=self.Spin_calc, kpts={'density': self.Ground_kpts_density, 'gamma': self.Gamma}, txt=self.struct+'-1-Log-Ground.txt',
                                convergence = self.Ground_convergence, occupations = self.Occupation)
                    else:
                        calc = GPAW(mode=PW(ecut=self.Cut_off_energy, force_complex_dtype=True), xc={'name': self.XC_calc, 'backend': 'pw'}, nbands='200%', 
                                parallel={'band': 1, 'kpt': 1}, eigensolver=Davidson(niter=1), mixer=self.Mixer_type, charge=self.Total_charge,
                                spinpol=self.Spin_calc, kpts={'size': (self.Ground_kpts_x, self.Ground_kpts_y, self.Ground_kpts_z), 'gamma': self.Gamma}, txt=self.struct+'-1-Log-Ground.txt',
                                convergence = self.Ground_convergence, occupations = self.Occupation)
                else:
                    parprint('Starting calculations with '+self.XC_calc+'...')
                    # Fix the spacegroup in the geometric optimization if wanted
                    if self.Fix_symmetry == True:
                        self.bulk_configuration.set_constraint(FixSymmetry(self.bulk_configuration))
                    if self.Ground_kpts_density is not None:
                        calc = GPAW(mode=PW(ecut=self.Cut_off_energy, force_complex_dtype=True), xc=self.XC_calc, nbands='200%', setups= self.Setup_params, 
                                parallel={'domain': world.size}, spinpol=self.Spin_calc, kpts={'density': self.Ground_kpts_density, 'gamma': self.Gamma},
                                mixer=self.Mixer_type, txt=self.struct+'-1-Log-Ground.txt', charge=self.Total_charge,
                                convergence = self.Ground_convergence, occupations = self.Occupation)
                    else:
                        calc = GPAW(mode=PW(ecut=self.Cut_off_energy, force_complex_dtype=True), xc=self.XC_calc, nbands='200%', setups= self.Setup_params, 
                                parallel={'domain': world.size}, spinpol=self.Spin_calc, kpts={'size': (self.Ground_kpts_x, self.Ground_kpts_y, self.Ground_kpts_z), 'gamma': self.Gamma},
                                mixer=self.Mixer_type, txt=self.struct+'-1-Log-Ground.txt', charge=self.Total_charge,
                                convergence = self.Ground_convergence, occupations = self.Occupation)
                self.bulk_configuration.calc = calc
                if self.Geo_optim == True:
                    if True in self.Relax_cell:
                        if self.Hydrostatic_pressure > 0.0:
                            uf = FrechetCellFilter(self.bulk_configuration, mask=self.Relax_cell, hydrostatic_strain=True, scalar_pressure=self.Hydrostatic_pressure)
                        else:
                            uf = FrechetCellFilter(self.bulk_configuration, mask=self.Relax_cell)
                        # Optimizer Selection
                        if self.Optimizer == 'FIRE':
                            from ase.optimize.fire import FIRE
                            relax = FIRE(uf, maxstep=self.Max_step, trajectory=self.struct+'-1-Result-Ground.traj')
                        elif  self.Optimizer == 'LBFGS':
                            from ase.optimize.lbfgs import LBFGS
                            relax = LBFGS(uf, maxstep=self.Max_step, alpha=self.Alpha, damping=self.Damping, trajectory=self.struct+'-1-Result-Ground.traj')
                        elif  self.Optimizer == 'GPMin':
                            from ase.optimize import GPMin
                            relax = GPMin(uf, trajectory=self.struct+'-1-Result-Ground.traj')
                        else:
                            relax = QuasiNewton(uf, maxstep=self.Max_step, trajectory=self.struct+'-1-Result-Ground.traj')
                    else:
                        # Optimizer Selection
                        if self.Optimizer == 'FIRE':
                            from ase.optimize.fire import FIRE
                            relax = FIRE(self.bulk_configuration, maxstep=self.Max_step, trajectory=self.struct+'-1-Result-Ground.traj')
                        elif  self.Optimizer == 'LBFGS':
                            from ase.optimize.lbfgs import LBFGS
                            relax = LBFGS(self.bulk_configuration, maxstep=self.Max_step, alpha=self.Alpha, damping=self.Damping, trajectory=self.struct+'-1-Result-Ground.traj')
                        elif  self.Optimizer == 'GPMin':
                            from ase.optimize import GPMin
                            relax = GPMin(self.bulk_configuration, trajectory=self.struct+'-1-Result-Ground.traj')
                        else:
                            relax = QuasiNewton(self.bulk_configuration, maxstep=self.Max_step, trajectory=self.struct+'-1-Result-Ground.traj')
                    relax.run(fmax=self.Max_F_tolerance)  # Consider tighter fmax!
                else:
                    self.bulk_configuration.set_calculator(calc)
                    self.bulk_configuration.get_potential_energy()
                #This line makes huge GPW files. Therefore, it is better to use this if-else
                calc.write(self.struct+'-1-Result-Ground.gpw', mode="all")

                # Writes final configuration as CIF file
                write_cif(self.struct+'-Final.cif', self.bulk_configuration)
            else:
                parprint("Passing PW ground state calculation...")
                # Control the ground state GPW file
                if not os.path.exists(self.struct+'-1-Result-Ground.gpw'):
                    parprint('\033[91mERROR:\033[0m'+self.struct+'-1-Result-Ground.gpw file can not be found. It is needed in other calculations. Firstly, finish the ground state calculation. You must have \033[95mGround_calc = True\033[0m line in your input file. Quiting.')
                    quit()

        elif self.Mode == 'LCAO':
            if self.Spin_calc == True:
                if self.Magmom_single_atom is not None:
                    numm = [0.0]*self.bulk_configuration.get_global_number_of_atoms()
                    numm[self.Magmom_single_atom[0]] = self.Magmom_single_atom[1]
                else:
                    numm = [self.Magmom_per_atom]*self.bulk_configuration.get_global_number_of_atoms()
                self.bulk_configuration.set_initial_magnetic_moments(numm)
            if self.Ground_calc == True:
                parprint("Starting LCAO ground state calculation...")
                # Fix the spacegroup in the geometric optimization if wanted
                if self.Fix_symmetry == True:
                    self.bulk_configuration.set_constraint(FixSymmetry(self.bulk_configuration))
                if self.Ground_gpts_density is not None:
                    if self.Ground_kpts_density is not None:
                        calc = GPAW(mode='lcao', basis='dzp', setups= self.Setup_params, kpts={'density': self.Ground_kpts_density, 'gamma': self.Gamma},
                                convergence = self.Ground_convergence, h=self.Ground_gpts_density, spinpol=self.Spin_calc, txt=self.struct+'-1-Log-Ground.txt',
                                mixer=self.Mixer_type, occupations = self.Occupation, nbands='200%', parallel={'domain': world.size}, charge=self.Total_charge)
                    else:
                        calc = GPAW(mode='lcao', basis='dzp', setups= self.Setup_params, kpts={'size':(self.Ground_kpts_x, self.Ground_kpts_y, self.Ground_kpts_z), 'gamma': self.Gamma},
                                convergence = self.Ground_convergence, h=self.Ground_gpts_density, spinpol=self.Spin_calc, txt=self.struct+'-1-Log-Ground.txt',
                                mixer=self.Mixer_type, occupations = self.Occupation, nbands='200%', parallel={'domain': world.size}, charge=self.Total_charge)
                else:
                    if self.Ground_kpts_density is not None:
                        calc = GPAW(mode='lcao', basis='dzp', setups= self.Setup_params, kpts={'density': self.Ground_kpts_density, 'gamma': self.Gamma},
                                convergence = self.Ground_convergence, gpts=(self.Ground_gpts_x, self.Ground_gpts_y, self.Ground_gpts_z), spinpol=self.Spin_calc, txt=self.struct+'-1-Log-Ground.txt',
                                mixer=self.Mixer_type, occupations = self.Occupation, nbands='200%', parallel={'domain': world.size}, charge=self.Total_charge)
                    else:
                        calc = GPAW(mode='lcao', basis='dzp', setups= self.Setup_params, kpts={'size':(self.Ground_kpts_x, self.Ground_kpts_y, self.Ground_kpts_z), 'gamma': self.Gamma},
                                convergence = self.Ground_convergence, gpts=(self.Ground_gpts_x, self.Ground_gpts_y, self.Ground_gpts_z), spinpol=self.Spin_calc, txt=self.struct+'-1-Log-Ground.txt',
                                mixer=self.Mixer_type, occupations = self.Occupation, nbands='200%', parallel={'domain': world.size}, charge=self.Total_charge)
                self.bulk_configuration.calc = calc
                if self.Geo_optim == True:
                    if True in self.Relax_cell:
                        #uf = FrechetCellFilter(self.bulk_configuration, mask=self.Relax_cell)
                        #relax = LBFGS(uf, maxstep=self.Max_step, alpha=self.Alpha, damping=self.Damping, trajectory=self.struct+'-1-Result-Ground.traj')
                        parprint('\033[91mERROR:\033[0mModifying supercell and atom positions with a filter (Relax_cell keyword) is not implemented in LCAO mode.')
                        quit()
                    else:
                        # Optimizer Selection
                        if self.Optimizer == 'FIRE':
                            from ase.optimize.fire import FIRE
                            relax = FIRE(self.bulk_configuration, maxstep=self.Max_step, trajectory=self.struct+'-1-Result-Ground.traj')
                        elif self.Optimizer == 'LBFGS':
                            from ase.optimize.lbfgs import LBFGS
                            relax = LBFGS(self.bulk_configuration, maxstep=self.Max_step, alpha=self.Alpha, damping=self.Damping, trajectory=self.struct+'-1-Result-Ground.traj')
                        elif self.Optimizer == 'GPMin':
                            from ase.optimize import GPMin
                            relax = GPMin(self.bulk_configuration, trajectory=self.struct+'-1-Result-Ground.traj')
                        else:
                            relax = QuasiNewton(self.bulk_configuration, maxstep=self.Max_step, trajectory=self.struct+'-1-Result-Ground.traj')
                    relax.run(fmax=self.Max_F_tolerance)  # Consider tighter fmax!
                else:
                    self.bulk_configuration.set_calculator(calc)
                    self.bulk_configuration.get_potential_energy()
                #relax = LBFGS(self.bulk_configuration, maxstep=self.Max_step, alpha=self.Alpha, damping=self.Damping, trajectory=self.struct+'-1-Result-Ground.traj')
                #relax.run(fmax=self.Max_F_tolerance)  # Consider much tighter fmax!
                #self.bulk_configuration.get_potential_energy()
                #This line makes huge GPW files. Therefore it is better to use this if else
                calc.write(self.struct+'-1-Result-Ground.gpw', mode="all")

                # Writes final configuration as CIF file
                write_cif(self.struct+'-Final.cif', self.bulk_configuration)
                # Print final spacegroup information
                parprint("Final Spacegroup:",get_spacegroup(self.bulk_configuration, symprec=1e-2))
            else:
                parprint("Passing LCAO ground state calculation...")
                # Control the ground state GPW file
                if not os.path.exists(self.struct+'-1-Result-Ground.gpw'):
                    parprint('\033[91mERROR:\033[0m'+self.struct+'-1-Result-Ground.gpw file can not be found. It is needed in other calculations. Firstly, finish the ground state calculation. You must have \033[95mGround_calc = True\033[0m line in your input file. Quiting.')
                    quit()

        elif self.Mode == 'FD':
            parprint("\033[91mERROR:\033[0mFD mode is not implemented in gpaw-tools yet...")
            quit()
        else:
            parprint("\033[91mERROR:\033[0mPlease enter correct mode information.")
            quit()
        # Finish ground state timing
        time12 = time.time()

        # Write timings of calculation
        with paropen(struct+'-7-Result-Log-Timings.txt', 'a') as f1:
            print('Ground state: ', round((time12-time11),2), end="\n", file=f1)

    def elasticcalc(self, drawfigs=False, strain_n=5, strain_mag=0.01, thickness=None):
        """
        Calculate the full elastic constant tensor and derived moduli.
        - strain_n: Number of strain points (including zero) for each independent strain mode.
        - strain_mag: Maximum strain magnitude (fractional, e.g., 0.01 for 1% strain).
        - thickness: Effective thickness for 2D materials (Angstrom).
        """
        # -------------------------------------------------------------
        # Step 1.5 - ELASTIC CALCULATION
        # -------------------------------------------------------------
        
        # Load the optimized (reference) structure
        bulk_atoms = self.bulk_configuration
        ref_calc = GPAW(self.struct + '-1-Result-Ground.gpw')
        bulk_atoms.set_calculator(ref_calc)
        parprint('Optimized (reference) structure is loaded.')
        try:
            ref_stress = bulk_atoms.get_stress(voigt=False)
        except Exception as e:
            raise RuntimeError(f"ERROR: Could not compute reference stress: {e}")
        parprint(f"Reference Stress Tensor (GPa):\n{ref_stress / GPa}")

        # --- Define the six independent strain matrices (Voigt components) ---
        strain_matrices = [
            np.array([[1, 0, 0],
                      [0, 0, 0],
                      [0, 0, 0]]),  # ε_xx
            np.array([[0, 0, 0],
                      [0, 1, 0],
                      [0, 0, 0]]),  # ε_yy
            np.array([[0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 1]]),  # ε_zz
            np.array([[0, 1, 0],
                      [1, 0, 0],
                      [0, 0, 0]]),  # ε_xy
            np.array([[0, 0, 1],
                      [0, 0, 0],
                      [1, 0, 0]]),  # ε_xz
            np.array([[0, 0, 0],
                      [0, 0, 1],
                      [0, 1, 0]])   # ε_yz
        ]

        # Define names for each strain mode (following Voigt notation)
        strain_names = ['ε_xx', 'ε_yy', 'ε_zz', 'ε_xy', 'ε_xz', 'ε_yz']
        # --- Cache file for deformed systems ---
        cache_file = self.struct + '1.5-Result-Elastic-deformations.traj'
        if os.path.exists(cache_file):
            parprint("Loading deformed systems from cache.")
            systems = read(cache_file, index=':')
        else:
            systems = []
            # For each strain mode, sample strain_n values linearly from -strain_mag to +strain_mag.
            for mode, strain in enumerate(strain_matrices):
                mode_label = strain_names[mode]
                strain_values = np.linspace(-strain_mag, strain_mag, strain_n)
                for eps in strain_values:
                    # If eps is effectively zero, use the reference ground state.
                    if abs(eps) < 1e-6:
                        parprint(f"Mode {mode_label}: Using ground-state stress for strain {eps:.4f}")
                        ref_atoms = bulk_atoms.copy()
                        # Use the previously computed reference values:
                        ref_atoms.set_calculator(SinglePointCalculator(ref_atoms, 
                                                energy=bulk_atoms.get_potential_energy(), 
                                                stress=ref_stress, 
                                                forces=bulk_atoms.get_forces()))
                        systems.append(ref_atoms)
                        continue
                    success = False
                    # Try a series of fallback factors to help convergence.
                    for factor in [1.0, 0.5, 0.1]:
                        current_eps = eps * factor
                        deformed_atoms = bulk_atoms.copy()
                        strain_tensor = np.eye(3) + current_eps * strain
                        deformed_atoms.set_cell(deformed_atoms.cell @ strain_tensor, scale_atoms=True)
            
                        # Attach a new GPAW calculator for the deformed structure using PBE.
                        deformed_atoms.set_calculator(GPAW(mode=PW(ecut=Cut_off_energy, force_complex_dtype=True), xc=XC_calc, 
                                    nbands='200%', setups= Setup_params, 
                                    parallel={'domain': world.size}, spinpol=Spin_calc, 
                                    kpts={'size': (Ground_kpts_x, Ground_kpts_y, Ground_kpts_z), 'gamma': Gamma},
                                    mixer=Mixer_type, txt=struct+'-1.5-Log-Elastic-deformations.txt', charge=Total_charge,
                                    convergence = Ground_convergence, occupations = Occupation))
                        try:
                            deformed_atoms.get_potential_energy()
                            # Force stress calculation.
                            deformed_atoms.calc.calculate(deformed_atoms, properties=['stress'])
                            stress_tensor = deformed_atoms.get_stress(voigt=False)
                            parprint(f"Mode {mode_label}: Computed stress for strain {eps:.4f} (factor {factor}):\n{stress_tensor}")
                            systems.append(deformed_atoms)
                            success = True
                            break
                        except Exception as e:
                            parprint(f"ERROR: Mode {mode_label}: Failed to compute stress for strain {eps:.4f} (factor {factor}): {e}")
                if not success:
                    parprint(f"WARNING: Mode {mode_label}: Skipping strain {eps:.4f} for current mode.")
            # Write deformed systems to cache file so that next time they can be loaded
            write(cache_file, systems)
            parprint(f"Deformed systems saved to cache file: {cache_file}")

        expected = len(strain_matrices) * strain_n
        if len(systems) < 0.5 * expected:
            raise RuntimeError(f"ERROR: Not enough valid deformations (only {len(systems)} out of {expected}) to compute the elastic tensor.")

        # --- Compute the elastic tensor using the deformed systems ---
        Cij, fit_info = get_elastic_tensor(bulk_atoms, systems)
        Cij = np.array(Cij)
        parprint(f"Cij raw shape: {Cij.shape}")
        if Cij.size < 36:
            Cij = reconstruct_full_tensor(Cij, bulk_atoms)
        elif Cij.size == 36:
            Cij = Cij.reshape((6,6))
        else:
            raise ValueError(f"ERROR: Unexpected Cij size: {Cij.size}. Cannot reshape.")
        Cij_GPa = Cij / GPa

        # --- 2D vs. 3D handling (unchanged) ---
        cell_lengths = bulk_atoms.cell.lengths()  # [a, b, c]
        is2D = False
        if thickness or (cell_lengths[2] > 3 * max(cell_lengths[0], cell_lengths[1]) and abs(Cij_GPa[2,2]) < 1e-2):
            is2D = True
            parprint("2D system is detected.")
            t_eff = thickness if thickness else cell_lengths[2]
            t_eff_m = t_eff * 1e-10  # convert Angstrom to meter
            C2D = Cij_GPa * (t_eff_m * 1e9)  # Convert GPa to N/m
            C11 = C2D[0, 0]; C22 = C2D[1, 1]; C12 = C2D[0, 1]
            E2D_x = (C11**2 - C12**2) / C11 if C11 > 0 else 0.0
            E2D = (E2D_x + ((C22**2 - C12**2) / C22 if C22 > 0 else 0.0)) / 2
            nu2D = C12 / C11 if C11 != 0 else 0.0
        else:
            parprint("2D system is not detected.")
            Bv = (Cij_GPa[0,0] + Cij_GPa[1,1] + Cij_GPa[2,2] +
                  2*(Cij_GPa[0,1] + Cij_GPa[0,2] + Cij_GPa[1,2])) / 9.0
            Gv = ((Cij_GPa[0,0] + Cij_GPa[1,1] + Cij_GPa[2,2]) -
                  (Cij_GPa[0,1] + Cij_GPa[0,2] + Cij_GPa[1,2]) +
                  3*(Cij_GPa[3,3] + Cij_GPa[4,4] + Cij_GPa[5,5])) / 15.0
            try:
                Sij = np.linalg.inv(Cij_GPa)
            except np.linalg.LinAlgError:
                Sij = np.linalg.pinv(Cij_GPa)
            Br = 1.0 / (Sij[0,0] + Sij[1,1] + Sij[2,2] +
                        2*(Sij[0,1] + Sij[0,2] + Sij[1,2]))
            Gr = 15.0 / (4*(Sij[0,0] + Sij[1,1] + Sij[2,2]) -
                         4*(Sij[0,1] + Sij[0,2] + Sij[1,2]) +
                         3*(Sij[3,3] + Sij[4,4] + Sij[5,5]))
            B_hill = 0.5 * (Bv + Br)
            G_hill = 0.5 * (Gv + Gr)
            E_hill = (9 * B_hill * G_hill) / (3 * B_hill + G_hill)
            nu_hill = (3 * B_hill - 2 * G_hill) / (2 * (3 * B_hill + G_hill))
        
        with paropen(self.struct + '-1.5-Result-Elastic-AllResults.txt', 'w') as fd:
            print("Elastic tensor Cij (GPa):", file=fd)
            print(np.array2string(Cij_GPa, precision=2, floatmode='fixed'), file=fd)
            if is2D:
                print(f"Detected 2D material. Effective thickness = {t_eff:.2f} Å.", file=fd)
                print("In-plane elastic stiffness (N/m):", file=fd)
                print(np.array2string(C2D, precision=2, floatmode='fixed'), file=fd)
                print(f"2D (in-plane) Young's modulus: {E2D:.2f} N/m", file=fd)
                print(f"2D Poisson's ratio (in-plane): {nu2D:.3f}", file=fd)
            else:
                print(f"Bulk modulus (Hill avg): {B_hill:.2f} GPa", file=fd)
                print(f"Shear modulus (Hill avg): {G_hill:.2f} GPa", file=fd)
                print(f"Young's modulus (Hill avg): {E_hill:.2f} GPa", file=fd)
                print(f"Poisson's ratio (Hill avg): {nu_hill:.3f}", file=fd)


    def doscalc(self, drawfigs = False):
        """
        This method performs density of states (DOS) calculations for the given structure using
        the ground state results. It computes the DOS for various energy levels and saves the
        results in appropriate files for further electronic analysis and visualization.
        """

        # -------------------------------------------------------------
        # Step 2 - DOS CALCULATION
        # -------------------------------------------------------------

        # Start DOS calc
        time21 = time.time()
        parprint("Starting DOS calculation...")
        if XC_calc in ['HSE06', 'HSE03','B3LYP', 'PBE0','EXX']:
            parprint('Passing DOS NSCF calculations...')
            calc = GPAW().read(filename=struct+'-1-Result-Ground.gpw')
            ef=0.0 # Can not find the use get_fermi_level() 
        else:
            calc = GPAW(struct+'-1-Result-Ground.gpw').fixed_density(txt=struct+'-2-Log-DOS.txt', convergence = DOS_convergence, occupations = Occupation)
            ef = calc.get_fermi_level()
        
        chem_sym = bulk_configuration.get_chemical_symbols()
        

        if Spin_calc == True:
            #Spin down

            # RAW PDOS for spin down
            parprint("Calculating and saving Raw PDOS for spin down...")
            if ef==0.0:
                rawdos = DOSCalculator.from_calculator(filename=struct+'-1-Result-Ground.gpw',soc=False, theta=0.0, phi=0.0, shift_fermi_level=True)
            else:
                rawdos = DOSCalculator.from_calculator(filename=struct+'-1-Result-Ground.gpw',soc=False, theta=0.0, phi=0.0, shift_fermi_level=False)
            energies = rawdos.get_energies(npoints=DOS_npoints)
            # Weights
            pdossweightsdown = [0.0] * DOS_npoints
            pdospweightsdown = [0.0] * DOS_npoints
            pdospxweightsdown = [0.0] * DOS_npoints
            pdospyweightsdown = [0.0] * DOS_npoints
            pdospzweightsdown = [0.0] * DOS_npoints
            pdosdweightsdown = [0.0] * DOS_npoints
            pdosdxyweightsdown = [0.0] * DOS_npoints
            pdosdyzweightsdown = [0.0] * DOS_npoints
            pdosd3z2_r2weightsdown = [0.0] * DOS_npoints
            pdosdzxweightsdown = [0.0] * DOS_npoints
            pdosdx2_y2weightsdown = [0.0] * DOS_npoints
            pdosfweightsdown = [0.0] * DOS_npoints
            totaldosweightsdown = [0.0] * DOS_npoints
            # Writing RawPDOS
            with paropen(struct+'-2-Result-RawPDOS-EachAtom-Down.csv', "w") as fd:
                print("Energy, s-total, p-total, px, py, pz, d-total, dxy, dyz, d3z2_r2, dzx, dx2_y2, f-total, TOTAL", file=fd)
                for j in range(0, bulk_configuration.get_global_number_of_atoms()):
                    print("Atom no: "+str(j+1)+", Atom Symbol: "+chem_sym[j]+" --------------------", file=fd)
                    pdoss = rawdos.raw_pdos(energies, a=j, l=0, m=None, spin=0, width=DOS_width)
                    pdosp = rawdos.raw_pdos(energies, a=j, l=1, m=None, spin=0, width=DOS_width)
                    pdospx = rawdos.raw_pdos(energies, a=j, l=1, m=2, spin=0, width=DOS_width)
                    pdospy = rawdos.raw_pdos(energies, a=j, l=1, m=0, spin=0, width=DOS_width)
                    pdospz = rawdos.raw_pdos(energies, a=j, l=1, m=1, spin=0, width=DOS_width)
                    pdosd = rawdos.raw_pdos(energies, a=j, l=2, m=None, spin=0, width=DOS_width)
                    pdosdxy = rawdos.raw_pdos(energies, a=j, l=2, m=0, spin=0, width=DOS_width)
                    pdosdyz = rawdos.raw_pdos(energies, a=j, l=2, m=1, spin=0, width=DOS_width)
                    pdosd3z2_r2 = rawdos.raw_pdos(energies, a=j, l=2, m=2, spin=0, width=DOS_width)
                    pdosdzx = rawdos.raw_pdos(energies, a=j, l=2, m=3, spin=0, width=DOS_width)
                    pdosdx2_y2 = rawdos.raw_pdos(energies, a=j, l=2, m=4, spin=0, width=DOS_width)
                    pdosf = rawdos.raw_pdos(energies, a=j, l=3, m=None, spin=0, width=DOS_width)
                    dosspdf = pdoss + pdosp + pdosd + pdosf
                    # Weights
                    pdossweightsdown = pdossweightsdown + pdoss
                    pdospweightsdown = pdospweightsdown + pdosp
                    pdospxweightsdown = pdospxweightsdown + pdospx
                    pdospyweightsdown = pdospyweightsdown + pdospy
                    pdospzweightsdown = pdospzweightsdown + pdospz
                    pdosdweightsdown = pdosdweightsdown + pdosd
                    pdosdxyweightsdown = pdosdxyweightsdown + pdosd
                    pdosdyzweightsdown = pdosdyzweightsdown + pdosd
                    pdosd3z2_r2weightsdown = pdosd3z2_r2weightsdown + pdosd
                    pdosdzxweightsdown = pdosdzxweightsdown + pdosd
                    pdosdx2_y2weightsdown = pdosdx2_y2weightsdown + pdosd
                    pdosfweightsdown = pdosfweightsdown + pdosf
                    totaldosweightsdown = totaldosweightsdown + dosspdf
                    for x in zip(energies, pdoss, pdosp, pdospx, pdospy, pdospz, pdosd, pdosdxy, pdosdyz, pdosd3z2_r2, pdosdzx, pdosdx2_y2, pdosf, dosspdf):
                        print(*x, sep=", ", file=fd)

            # Writing DOS
            parprint("Saving DOS for spin down...")
            with paropen(struct+'-2-Result-DOS-Down.csv', "w") as fd:
                for x in zip(energies, totaldosweightsdown):
                    print(*x, sep=", ", file=fd)
                    
            # Writing PDOS
            parprint("Saving PDOS for spin down...")
            with paropen(struct+'-2-Result-PDOS-Down.csv', "w") as fd:
                print("Energy, s-total, p-total, px, py, pz, d-total, dxy, dyz, d3z2_r2, dzx, dx2_y2, f-total, TOTAL", file=fd)
                for x in zip(energies, pdossweightsdown, pdospweightsdown, pdospxweightsdown, pdospyweightsdown, pdospzweightsdown, pdosdweightsdown,
                             pdosdxyweightsdown, pdosdyzweightsdown, pdosd3z2_r2weightsdown, pdosdzxweightsdown, pdosdx2_y2weightsdown, pdosfweightsdown, totaldosweightsdown):
                    print(*x, sep=", ", file=fd)

            #Spin up

            # RAW PDOS for spin up
            parprint("Calculating and saving Raw PDOS for spin up...")
            rawdos = DOSCalculator.from_calculator(struct+'-1-Result-Ground.gpw',soc=False, theta=0.0, phi=0.0, shift_fermi_level=True)
            energies = rawdos.get_energies(npoints=DOS_npoints)
            # Weights
            pdossweightsup = [0.0] * DOS_npoints
            pdospweightsup = [0.0] * DOS_npoints
            pdospxweightsup = [0.0] * DOS_npoints
            pdospyweightsup = [0.0] * DOS_npoints
            pdospzweightsup = [0.0] * DOS_npoints
            pdosdweightsup = [0.0] * DOS_npoints
            pdosdxyweightsup = [0.0] * DOS_npoints
            pdosdyzweightsup = [0.0] * DOS_npoints
            pdosd3z2_r2weightsup = [0.0] * DOS_npoints
            pdosdzxweightsup = [0.0] * DOS_npoints
            pdosdx2_y2weightsup = [0.0] * DOS_npoints
            pdosfweightsup = [0.0] * DOS_npoints
            totaldosweightsup = [0.0] * DOS_npoints

            #Writing RawPDOS
            with paropen(struct+'-2-Result-RawPDOS-EachAtom-Up.csv', "w") as fd:
                print("Energy, s-total, p-total, px, py, pz, d-total, dxy, dyz, d3z2_r2, dzx, dx2_y2, f-total, TOTAL", file=fd)
                for j in range(0, bulk_configuration.get_global_number_of_atoms()):
                    print("Atom no: "+str(j+1)+", Atom Symbol: "+chem_sym[j]+" --------------------", file=fd)
                    pdoss = rawdos.raw_pdos(energies, a=j, l=0, m=None, spin=1, width=DOS_width)
                    pdosp = rawdos.raw_pdos(energies, a=j, l=1, m=None, spin=1, width=DOS_width)
                    pdospx = rawdos.raw_pdos(energies, a=j, l=1, m=2, spin=1, width=DOS_width)
                    pdospy = rawdos.raw_pdos(energies, a=j, l=1, m=0, spin=1, width=DOS_width)
                    pdospz = rawdos.raw_pdos(energies, a=j, l=1, m=1, spin=1, width=DOS_width)
                    pdosd = rawdos.raw_pdos(energies, a=j, l=2, m=None, spin=1, width=DOS_width)
                    pdosdxy = rawdos.raw_pdos(energies, a=j, l=2, m=0, spin=1, width=DOS_width)
                    pdosdyz = rawdos.raw_pdos(energies, a=j, l=2, m=1, spin=1, width=DOS_width)
                    pdosd3z2_r2 = rawdos.raw_pdos(energies, a=j, l=2, m=2, spin=1, width=DOS_width)
                    pdosdzx = rawdos.raw_pdos(energies, a=j, l=2, m=3, spin=1, width=DOS_width)
                    pdosdx2_y2 = rawdos.raw_pdos(energies, a=j, l=2, m=4, spin=1, width=DOS_width)
                    pdosf = rawdos.raw_pdos(energies, a=j, l=3, m=None, spin=1, width=DOS_width)
                    dosspdf = pdoss + pdosp + pdosd + pdosf
                    # Weights
                    pdossweightsup = pdossweightsup + pdoss
                    pdospweightsup = pdospweightsup + pdosp
                    pdospxweightsup = pdospxweightsup + pdospx
                    pdospyweightsup = pdospyweightsup + pdospy
                    pdospzweightsup = pdospzweightsup + pdospz
                    pdosdweightsup = pdosdweightsup + pdosd
                    pdosdxyweightsup = pdosdxyweightsup + pdosd
                    pdosdyzweightsup = pdosdyzweightsup + pdosd
                    pdosd3z2_r2weightsup = pdosd3z2_r2weightsup + pdosd
                    pdosdzxweightsup = pdosdzxweightsup + pdosd
                    pdosdx2_y2weightsup = pdosdx2_y2weightsup + pdosd
                    pdosfweightsup = pdosfweightsup + pdosf
                    totaldosweightsup = totaldosweightsup + dosspdf
                    for x in zip(energies, pdoss, pdosp, pdospx, pdospy, pdospz, pdosd, pdosdxy, pdosdyz, pdosd3z2_r2, pdosdzx, pdosdx2_y2, pdosf, dosspdf):
                        print(*x, sep=", ", file=fd)

            # Writing DOS
            parprint("Saving DOS for spin up...")
            with paropen(struct+'-2-Result-DOS-Up.csv', "w") as fd:
                for x in zip(energies, totaldosweightsup):
                    print(*x, sep=", ", file=fd)
            
            # Writing PDOS
            parprint("Saving PDOS for spin up...")
            with paropen(struct+'-2-Result-PDOS-Up.csv', "w") as fd:
                print("Energy, s-total, p-total, px, py, pz, d-total, dxy, dyz, d3z2_r2, dzx, dx2_y2, f-total, TOTAL", file=fd)
                for x in zip(energies, pdossweightsup, pdospweightsup, pdospxweightsup, pdospyweightsup, pdospzweightsup, pdosdweightsup, pdosdxyweightsup, 
                             pdosdyzweightsup, pdosd3z2_r2weightsup, pdosdzxweightsup, pdosdx2_y2weightsup, pdosfweightsup, totaldosweightsup):
                    print(*x, sep=", ", file=fd)

        else:

            # RAW PDOS
            parprint("Calculating and saving Raw PDOS...")
            rawdos = DOSCalculator.from_calculator(struct+'-1-Result-Ground.gpw',soc=False, theta=0.0, phi=0.0, shift_fermi_level=True)
            energies = rawdos.get_energies(npoints=DOS_npoints)
            totaldosweights = [0.0] * DOS_npoints
            pdossweights = [0.0] * DOS_npoints
            pdospweights = [0.0] * DOS_npoints
            pdospxweights = [0.0] * DOS_npoints
            pdospyweights = [0.0] * DOS_npoints
            pdospzweights = [0.0] * DOS_npoints
            pdosdweights = [0.0] * DOS_npoints
            pdosdxyweights = [0.0] * DOS_npoints
            pdosdyzweights = [0.0] * DOS_npoints
            pdosd3z2_r2weights = [0.0] * DOS_npoints
            pdosdzxweights = [0.0] * DOS_npoints
            pdosdx2_y2weights = [0.0] * DOS_npoints
            pdosfweights = [0.0] * DOS_npoints

            # Writing RawPDOS
            with paropen(struct+'-2-Result-RawPDOS-EachAtom.csv', "w") as fd:
                print("Energy, s-total, p-total, px, py, pz, d-total, dxy, dyz, d3z2_r2, dzx, dx2_y2, f-total, TOTAL", file=fd)
                for j in range(0, bulk_configuration.get_global_number_of_atoms()):
                    print("Atom no: "+str(j+1)+", Atom Symbol: "+chem_sym[j]+" ----------------------------------------", file=fd)
                    pdoss = rawdos.raw_pdos(energies, a=j, l=0, m=None, spin=None, width=DOS_width)
                    pdosp = rawdos.raw_pdos(energies, a=j, l=1, m=None, spin=None, width=DOS_width)
                    pdospx = rawdos.raw_pdos(energies, a=j, l=1, m=2, spin=None, width=DOS_width)
                    pdospy = rawdos.raw_pdos(energies, a=j, l=1, m=0, spin=None, width=DOS_width)
                    pdospz = rawdos.raw_pdos(energies, a=j, l=1, m=1, spin=None, width=DOS_width)
                    pdosd = rawdos.raw_pdos(energies, a=j, l=2, m=None, spin=None, width=DOS_width)
                    pdosdxy = rawdos.raw_pdos(energies, a=j, l=2, m=0, spin=None, width=DOS_width)
                    pdosdyz = rawdos.raw_pdos(energies, a=j, l=2, m=1, spin=None, width=DOS_width)
                    pdosd3z2_r2 = rawdos.raw_pdos(energies, a=j, l=2, m=2, spin=None, width=DOS_width)
                    pdosdzx = rawdos.raw_pdos(energies, a=j, l=2, m=3, spin=None, width=DOS_width)
                    pdosdx2_y2 = rawdos.raw_pdos(energies, a=j, l=2, m=4, spin=None, width=DOS_width)
                    pdosf = rawdos.raw_pdos(energies, a=j, l=3, m=None, spin=None, width=DOS_width)
                    # Weights
                    dosspdf = pdoss + pdosp + pdosd + pdosf
                    pdossweights = pdossweights + pdoss
                    pdospweights = pdospweights + pdosp
                    pdospxweights = pdospxweights + pdospx
                    pdospyweights = pdospyweights + pdospy
                    pdospzweights = pdospzweights + pdospz
                    pdosdweights = pdosdweights + pdosd
                    pdosdxyweights = pdosdxyweights + pdosd
                    pdosdyzweights = pdosdyzweights + pdosd
                    pdosd3z2_r2weights = pdosd3z2_r2weights + pdosd
                    pdosdzxweights = pdosdzxweights + pdosd
                    pdosdx2_y2weights = pdosdx2_y2weights + pdosd
                    pdosfweights = pdosfweights + pdosf
                    totaldosweights = totaldosweights + dosspdf
                    for x in zip(energies, pdoss, pdosp, pdospx, pdospy, pdospz, pdosd, pdosdxy, pdosdyz, pdosd3z2_r2, pdosdzx, pdosdx2_y2, pdosf, dosspdf):
                        print(*x, sep=", ", file=fd)

            # Writing DOS
            parprint("Saving DOS...")
            with paropen(struct+'-2-Result-DOS.csv', "w") as fd:
                for x in zip(energies, totaldosweights):
                    print(*x, sep=", ", file=fd)
            
            # Writing PDOS
            parprint("Saving PDOS...")
            with paropen(struct+'-2-Result-PDOS.csv', "w") as fd:
                print("Energy, s-total, p-total, px, py, pz, d-total, dxy, dyz, d3z2_r2, dzx, dx2_y2, f-total, TOTAL", file=fd)
                for x in zip(energies, pdossweights, pdospweights, pdospxweights, pdospyweights, pdospzweights, pdosdweights, pdosdxyweights, pdosdyzweights, 
                             pdosd3z2_r2weights, pdosdzxweights, pdosdx2_y2weights, pdosfweights, totaldosweights):
                    print(*x, sep=", ", file=fd)
                    
        # Finish DOS calc
        time22 = time.time()
        # Write timings of calculation
        with paropen(struct+'-7-Result-Log-Timings.txt', 'a') as f1:
            print('DOS calculation: ', round((time22-time21),2), end="\n", file=f1)

        # Write or draw figures
        if drawfigs == True:
            # Draw graphs only on master node
            if world.rank == 0:
                # DOS
                if Spin_calc == True:
                    downf = pd.read_csv(struct+'-2-Result-DOS-Down.csv', header=None)
                    upf = pd.read_csv(struct+'-2-Result-DOS-Up.csv', header=None)
                    downf[0]=downf[0]+ef
                    upf[0]=upf[0]+ef
                    ax = plt.gca()
                    ax.plot(downf[0], -1.0*downf[1], 'y')
                    ax.plot(upf[0], upf[1], 'b')
                    ax.set_xlabel(dos_xlabel[Localisation])
                    ax.set_ylabel(dos_ylabel[Localisation])
                else:
                    dosf = pd.read_csv(struct+'-2-Result-DOS.csv', header=None)
                    dosf[0]=dosf[0]+ef
                    ax = plt.gca()
                    ax.plot(dosf[0], dosf[1], 'b')
                    ax.set_xlabel(dos_xlabel[Localisation])
                    ax.set_ylabel(dos_ylabel[Localisation])
                plt.xlim(Energy_min+ef, Energy_max+ef)
                autoscale_y(ax)
                plt.savefig(struct+'-2-Graph-DOS.png', dpi=300)
                #plt.show()
        else:
            # Draw graphs only on master node
            if world.rank == 0:
                # DOS
                if Spin_calc == True:
                    downf = pd.read_csv(struct+'-2-Result-DOS-Down.csv', header=None)
                    upf = pd.read_csv(struct+'-2-Result-DOS-Up.csv', header=None)
                    downf[0]=downf[0]+ef
                    upf[0]=upf[0]+ef
                    ax = plt.gca()
                    ax.plot(downf[0], -1.0*downf[1], 'y')
                    ax.plot(upf[0], upf[1], 'b')
                    ax.set_xlabel(dos_xlabel[Localisation])
                    ax.set_ylabel(dos_ylabel[Localisation])
                else:
                    dosf = pd.read_csv(struct+'-2-Result-DOS.csv', header=None)
                    dosf[0]=dosf[0]+ef
                    ax = plt.gca()
                    ax.plot(dosf[0], dosf[1], 'b')
                    ax.set_xlabel(dos_xlabel[Localisation])
                    ax.set_ylabel(dos_ylabel[Localisation])
                plt.xlim(Energy_min+ef, Energy_max+ef)
                autoscale_y(ax)
                plt.savefig(struct+'-2-Graph-DOS.png', dpi=300)

    def bandcalc(self, drawfigs = False):
        """
        This method performs band structure calculations for the given structure using the
        ground state results. It computes the electronic band structure along specified
        k-point paths and saves the results in appropriate files for further analysis
        and visualization.
        """

        # -------------------------------------------------------------
        # Step 3 - BAND STRUCTURE CALCULATION
        # -------------------------------------------------------------

        # Start Band calc
        time31 = time.time()
        parprint("Starting band structure calculation...")
        if XC_calc in ['HSE06', 'HSE03','B3LYP', 'PBE0','EXX']:
            calc = GPAW(struct+'-1-Result-Ground.gpw', symmetry='off',kpts={'path': Band_path, 'npoints': Band_npoints},
                      parallel={'band':1, 'kpt':1}, occupations = Occupation,
                      txt=struct+'-3-Log-Band.txt', convergence=Band_convergence)
            ef=0.0

        else:
            calc = GPAW(struct+'-1-Result-Ground.gpw').fixed_density(kpts={'path': Band_path, 'npoints': Band_npoints},
                      txt=struct+'-3-Log-Band.txt', symmetry='off', occupations = Occupation, convergence=Band_convergence)
            ef = calc.get_fermi_level()

        calc.get_potential_energy()
        bs = calc.band_structure()
            
        Band_num_of_bands = calc.get_number_of_bands()
        parprint('Num of bands:'+str(Band_num_of_bands))

        # No need to write an additional gpaw file. Use json file to use with ase band-structure command
        #calc.write(struct+'-3-Result-Band.gpw')
        bs.write(struct+'-3-Result-Band.json')

        if Spin_calc == True:
            eps_skn = np.array([[calc.get_eigenvalues(k,s)
                                for k in range(Band_npoints)]
                                for s in range(2)]) - ef
            parprint(eps_skn.shape)
            with paropen(struct+'-3-Result-Band-Down.dat', 'w') as f1:
                for n1 in range(Band_num_of_bands):
                    for k1 in range(Band_npoints):
                        print(k1, eps_skn[0, k1, n1], end="\n", file=f1)
                    print (end="\n", file=f1)

            with paropen(struct+'-3-Result-Band-Up.dat', 'w') as f2:
                for n2 in range(Band_num_of_bands):
                    for k2 in range(Band_npoints):
                        print(k2, eps_skn[1, k2, n2], end="\n", file=f2)
                    print (end="\n", file=f2)

            # Thanks to Andrej Kesely (https://stackoverflow.com/users/10035985/andrej-kesely) for helping the problem of general XYYY writer
            currentd, all_groupsd = [], []
            with open(struct+'-3-Result-Band-Down.dat', 'r') as f_in1:
                for line in map(str.strip, f_in1):
                    if line == "" and currentd:
                        all_groupsd.append(currentd)
                        currentd = []
                    else:
                        currentd.append(line.split(maxsplit=1))

            if currentd:
                all_groupsd.append(currentd)

            try:
                with paropen(struct+'-3-Result-Band-Down-XYYY.dat', 'w') as f1:
                    for g in zip(*all_groupsd):
                        print('{} {} {}'.format(g[0][0], g[0][1], ' '.join(v for _, v in g[1:])), file=f1)
            except Exception as e:
                print("\033[93mWARNING:\033[0m A problem occurred during writing XYYY formatted spin down Band file. Mostly, the file is created without any problem.")
                print(e)
                pass  # Continue execution after encountering an exception

            currentu, all_groupsu = [], []
            with open(struct+'-3-Result-Band-Up.dat', 'r') as f_in2:
                for line in map(str.strip, f_in2):
                    if line == "" and currentu:
                        all_groupsu.append(currentu)
                        currentu = []
                    else:
                    currentu.append(line.split(maxsplit=1))

            if currentu:
                all_groupsu.append(currentu)
            try:
                with paropen(struct+'-3-Result-Band-Up-XYYY.dat', 'w') as f2:
                    for g in zip(*all_groupsu):
                        print('{} {} {}'.format(g[0][0], g[0][1], ' '.join(v for _, v in g[1:])), file=f2)
            except Exception as e:
                print("\033[93mWARNING:\033[0m A problem occurred during writing XYYY formatted spin up Band file. Mostly, the file is created without any problem.")
                print(e)
                pass  # Continue execution after encountering an exception

        else:
            eps_skn = np.array([[calc.get_eigenvalues(k,s)
                                for k in range(Band_npoints)]
                                for s in range(1)]) - ef
            with paropen(struct+'-3-Result-Band.dat', 'w') as f:
                for n in range(Band_num_of_bands):
                    for k in range(Band_npoints):
                        print(k, eps_skn[0, k, n], end="\n", file=f)
                    print (end="\n", file=f)

            # Thanks to Andrej Kesely (https://stackoverflow.com/users/10035985/andrej-kesely) for helping the problem of general XYYY writer
            current, all_groups = [], []
            with open(struct+'-3-Result-Band.dat', 'r') as f_in:
                for line in map(str.strip, f_in):
                    if line == "" and current:
                        all_groups.append(current)
                        current = []
                    else:
                        current.append(line.split(maxsplit=1))

            if current:
                all_groups.append(current)
            try:
                with paropen(struct+'-3-Result-Band-XYYY.dat', 'w') as f1:
                    for g in zip(*all_groups):
                        print('{} {} {}'.format(g[0][0], g[0][1], ' '.join(v for _, v in g[1:])), file=f1)
            except Exception as e:
                print("\033[93mWARNING:\033[0m A problem occurred during writing XYYY formatted Band file. Mostly, the file is created without any problem.")
                print(e)
                pass  # Continue execution after encountering an exception
                
            # Projected Band
            Projected_band = False
            if Projected_band == True:                
                with paropen(struct+'-3-Result-ProjectedBand.dat', 'w') as f3:
                    for i in range(len(sym_ang_mom_i)):
                        print('----------------------'+sym_ang_mom_i[i]+'---------------------------', end="\n", file=f3)
                        for n in range(Band_num_of_bands):
                            for k in range(Band_npoints):
                                print(k, projector_weight_skni[0, k, n, i], end="\n", file=f3)
                            print (end="\n", file=f3)
                
        # Finish Band calc
        time32 = time.time()
        # Write timings of calculation
        with paropen(struct+'-7-Result-Log-Timings.txt', 'a') as f1:
            print('Band calculation: ', round((time32-time31),2), end="\n", file=f1)

        # Write or draw figures
        if drawfigs == True:
            # Draw graphs only on master node
            if world.rank == 0:
                # Band Structure
                bs.plot(filename=struct+'-3-Graph-Band.png', show=True, emax=Energy_max + bs.reference, emin=Energy_min + bs.reference, ylabel=band_ylabel[Localisation])
        else:
            # Draw graphs only on master node
            if world.rank == 0:
                # Band Structure
                bs.plot(filename=struct+'-3-Graph-Band.png', show=False, emax=Energy_max + bs.reference, emin=Energy_min + bs.reference, ylabel=band_ylabel[Localisation])

    def densitycalc(self):
        """
        This method performs density calculations for the given structure using the
        ground state results. It computes the electron density distribution and saves
        the results in appropriate files for further analysis and visualization.
        """

        # -------------------------------------------------------------
        # Step 4 - ALL-ELECTRON DENSITY
        # -------------------------------------------------------------

        #Start Density calc
        time41 = time.time()
        parprint("Starting All-electron density calculation...")
        calc = GPAW(struct+'-1-Result-Ground.gpw', txt=struct+'-4-Log-ElectronDensity.txt')
        bulk_configuration.calc = calc
        if Spin_calc == True:
            np = calc.get_pseudo_density()
            n = calc.get_all_electron_density(gridrefinement=Refine_grid)
            # For spins
            npdown = calc.get_pseudo_density(spin=0)
            ndown = calc.get_all_electron_density(spin=0, gridrefinement=Refine_grid)
            npup = calc.get_pseudo_density(spin=1)
            nup = calc.get_all_electron_density(spin=1, gridrefinement=Refine_grid)
            # Zeta
            nzeta = (nup - ndown) / (nup + ndown)
            npzeta = (npup - npdown) / (npup + npdown)
            # Writing spin down pseudo and all electron densities to cube file with Bohr unit
            write(struct+'-4-Result-All-electron_nall-Down.cube', bulk_configuration, data=ndown * Bohr**3)
            write(struct+'-4-Result-All-electron_npseudo-Down.cube', bulk_configuration, data=npdown * Bohr**3)
            # Writing spin up pseudo and all electron densities to cube file with Bohr unit
            write(struct+'-4-Result-All-electron_nall-Up.cube', bulk_configuration, data=nup * Bohr**3)
            write(struct+'-4-Result-All-electron_npseudo-Up.cube', bulk_configuration, data=npup * Bohr**3)
            # Writing total pseudo and all electron densities to cube file with Bohr unit
            write(struct+'-4-Result-All-electron_nall-Total.cube', bulk_configuration, data=n * Bohr**3)
            write(struct+'-4-Result-All-electron_npseudo-Total.cube', bulk_configuration, data=np * Bohr**3)
            # Writing zeta pseudo and all electron densities to cube file with Bohr unit
            write(struct+'-4-Result-All-electron_nall-Zeta.cube', bulk_configuration, data=nzeta * Bohr**3)
            write(struct+'-4-Result-All-electron_npseudo-Zeta.cube', bulk_configuration, data=npzeta * Bohr**3)
        else:
            np = calc.get_pseudo_density()
            n = calc.get_all_electron_density(gridrefinement=Refine_grid)
            # Writing pseudo and all electron densities to cube file with Bohr unit
            write(struct+'-4-Result-All-electron_nall-Total.cube', bulk_configuration, data=n * Bohr**3)
            write(struct+'-4-Result-All-electron_npseudo-Total.cube', bulk_configuration, data=np * Bohr**3)
            
        # Finish Density calc
        time42 = time.time()
        # Write timings of calculation
        with paropen(struct+'-7-Result-Log-Timings.txt', 'a') as f1:
            print('Density calculation: ', round((time42-time41),2), end="\n", file=f1)

    def phononcalc(self):
        """
        This method performs a phonon calculation for the given structure using the ground state results. 
        It generates atomic displacements, computes force constants, and calculates phonon dispersion and phonon DOS.
        The results are saved as PNG file for now.
        """
        # -------------------------------------------------------------
        # Step 5 - PHONON CALCULATION
        # -------------------------------------------------------------

        time51 = time.time()
        parprint("Starting phonon calculation.(\033[93mWARNING:\033[0mNOT TESTED FEATURE, PLEASE CONTROL THE RESULTS)")

        calc = GPAW(struct+'-1-Result-Ground.gpw')
        bulk_configuration.calc = calc

        # Pre-process
        bulk_configuration_ph = convert_atoms_to_phonopy(bulk_configuration)
        phonon = Phonopy(bulk_configuration_ph, Phonon_supercell, log_level=1)
        phonon.generate_displacements(distance=Phonon_displacement)
        with paropen(struct+'-5-Log-Phonon-Phonopy.txt', 'a') as f2:
            print("[Phonopy] Atomic displacements:", end="\n", file=f2)
            disps = phonon.displacements
            for d in disps:
                print("[Phonopy] %d %s" % (d[0], d[1:]), end="\n", file=f2)

        # FIX THIS PART
        calc = GPAW(mode=PW(Phonon_PW_cutoff),
               kpts={'size': (Phonon_kpts_x, Phonon_kpts_y, Phonon_kpts_z)}, txt=struct+'-5-Log-Phonon-GPAW.txt')

        bulk_configuration.calc = calc

        path = get_band_path(bulk_configuration, Phonon_path, Phonon_npoints)

        phonon_path = struct+'5-Results-force-constants.npy'
        sum_rule=Phonon_acoustic_sum_rule

        if os.path.exists(phonon_path):
            with paropen(struct+'-5-Log-Phonon-Phonopy.txt', 'a') as f2:
                print('Reading FCs from {!r}'.format(phonon_path), end="\n", file=f2)
            phonon.force_constants = np.load(phonon_path)

        else:
            with paropen(struct+'-5-Log-Phonon-Phonopy.txt', 'a') as f2:
                print('Computing FCs',end="\n", file=f2)
                #os.makedirs('force-sets', exist_ok=True)
            supercells = list(phonon.supercells_with_displacements)
            fnames = [struct+'5-Results-sc-{:04}.npy'.format(i) for i in range(len(supercells))]
            set_of_forces = [
                load_or_compute_force(fname, calc, supercell)
                for (fname, supercell) in zip(fnames, supercells)
            ]
            with paropen(struct+'-5-Log-Phonon-Phonopy.txt', 'a') as f2:
                print('Building FC matrix', end="\n", file=f2)
            phonon.produce_force_constants(forces=set_of_forces, calculate_full_force_constants=False)
            if sum_rule:
                phonon.symmetrize_force_constants()
            with paropen(struct+'-5-Log-Phonon-Phonopy.txt', 'a') as f2:
                print('Writing FCs to {!r}'.format(phonon_path), end="\n", file=f2)
            np.save(phonon_path, phonon.force_constants)
            #shutil.rmtree('force-sets')

        with paropen(struct+'-5-Log-Phonon-Phonopy.txt', 'a') as f2:
            print('', end="\n", file=f2)
            print("[Phonopy] Phonon frequencies at Gamma:", end="\n", file=f2)
            for i, freq in enumerate(phonon.get_frequencies((0, 0, 0))):
                print("[Phonopy] %3d: %10.5f THz" %  (i + 1, freq), end="\n", file=f2) # THz

            # DOS
            print("[Phonopy] Initializing mesh [21, 21, 21]...", end="\n", file=f2)
            phonon.init_mesh([21, 21, 21])
            print("[Phonopy] Running total DOS calculation...", end="\n", file=f2)
            phonon.run_total_dos()
            print("[Phonopy] DOS calculation completed. Type of total_dos:", type(phonon.total_dos), end="\n", file=f2)
            if phonon.total_dos is not None:
                dos_array = np.array(phonon.total_dos)
                print("[Phonopy] DOS array shape:", dos_array.shape, "ndim:", dos_array.ndim, end="\n", file=f2)
            print('', end="\n", file=f2)
            print("[Phonopy] Phonon DOS:", end="\n", file=f2)
            
            # Check if total_dos is properly calculated and has the expected structure
            try:
                if phonon.total_dos is not None:
                    # Check if total_dos is a TotalDos object (new Phonopy format)
                    if hasattr(phonon.total_dos, 'frequency_points') and hasattr(phonon.total_dos, 'dos'):
                        frequencies = phonon.total_dos.frequency_points
                        dos_values = phonon.total_dos.dos
                        print("[Phonopy] Found TotalDos object with %d frequency points" % len(frequencies), end="\n", file=f2)
                        for omega, dos in zip(frequencies, dos_values):
                            print("%15.7f%15.7f" % (omega, dos), end="\n", file=f2)
                    # Check if total_dos is a tuple (freq, dos) as in older Phonopy versions
                    elif isinstance(phonon.total_dos, (tuple, list)) and len(phonon.total_dos) == 2:
                        frequencies, dos_values = phonon.total_dos
                        freq_array = np.array(frequencies)
                        dos_array = np.array(dos_values)
                        if freq_array.ndim == 1 and dos_array.ndim == 1 and len(freq_array) == len(dos_array):
                            for omega, dos in zip(freq_array, dos_array):
                                print("%15.7f%15.7f" % (omega, dos), end="\n", file=f2)
                        else:
                            print("[Phonopy] Warning: DOS frequency/values arrays have incompatible shapes", end="\n", file=f2)
                            print("[Phonopy] freq shape: %s, dos shape: %s" % (freq_array.shape, dos_array.shape), end="\n", file=f2)
                    else:
                        # Try the old method in case the format is different
                        dos_array = np.array(phonon.total_dos)
                        print("[Phonopy] DOS array shape:", dos_array.shape, "ndim:", dos_array.ndim, end="\n", file=f2)
                        if dos_array.ndim >= 2:  # Check if it's at least 2D
                            for omega, dos in dos_array.T:
                                print("%15.7f%15.7f" % (omega, dos), end="\n", file=f2)
                        else:
                            print("[Phonopy] Warning: DOS data has unexpected structure (ndim=%d)" % dos_array.ndim, end="\n", file=f2)
                            print("[Phonopy] DOS calculation may have failed or incomplete", end="\n", file=f2)
                            print("[Phonopy] Available attributes: %s" % [attr for attr in dir(phonon.total_dos) if not attr.startswith('_')], end="\n", file=f2)
                else:
                    print("[Phonopy] Warning: DOS data is None, calculation may have failed", end="\n", file=f2)
            except Exception as e:
                print("[Phonopy] Error processing DOS data: %s" % str(e), end="\n", file=f2)
                print("[Phonopy] Skipping DOS output to log file", end="\n", file=f2)

        qpoints, labels, connections = path
        phonon.run_band_structure(qpoints, path_connections=connections, labels=labels)

        # without DOS
        # fig = phonon.plot_band_structure()

        # with DOS
        phonon.run_mesh([20, 20, 20])
        phonon.run_total_dos()
        fig = phonon.plot_band_structure_and_dos()

        # with PDOS
        # phonon.run_mesh([20, 20, 20], with_eigenvectors=True, is_mesh_symmetry=False)
        # fig = phonon.plot_band_structure_and_dos(pdoc_indices=[[0], [1]])

        fig.savefig(struct+'-5-Result-Phonon.png', dpi=300)

        time52 = time.time()
        # Write timings of calculation
        with paropen(struct+'-7-Result-Log-Timings.txt', 'a') as f1:
            print('Phonon calculation: ', round((time52-time51),2), end="\n", file=f1)

    def opticalcalc(self):
        """
        This method performs optical property calculations for the given structure using the
        ground state results. It computes the dielectric function, absorption spectrum, and
        other related optical properties. The results are saved in appropriate files for
        further analysis and visualization.
        """

        # -------------------------------------------------------------
        # Step 6 - OPTICAL CALCULATION
        # -------------------------------------------------------------

        #Start Optical calc
        time61 = time.time()
        if Mode == 'PW':
            parprint("Starting optical calculation...")
            try:
                calc = GPAW(struct+'-1-Result-Ground.gpw').fixed_density(txt=struct+'-6-Log-Optical.txt',
                        nbands=Opt_num_of_bands,parallel={'domain': 1, 'band': 1 },
                        occupations=FermiDirac(Opt_FD_smearing))
            except FileNotFoundError as err:
                # output error, and return with an error code
                parprint('\033[91mERROR:\033[0mOptical computations must be done separately. Please do ground calculations first.')
                quit()

            calc.get_potential_energy()

            calc.diagonalize_full_hamiltonian(nbands=Opt_num_of_bands)  # diagonalize Hamiltonian
            calc.write(struct+'-6-Result-Optical.gpw', mode= 'all')  # write wavefunctions

            #from mpi4py import MPI
            if Opt_calc_type == 'BSE':
                if Spin_calc == True:
                   parprint('\033[91mERROR:\033[0mBSE calculations can not run with spin dependent data.')
                   quit()
                parprint('Starting BSE calculations')
                bse = BSE(calc= struct+'-6-Result-Optical.gpw', ecut=Opt_cut_of_energy,
                             valence_bands=Opt_BSE_valence,
                             conduction_bands=Opt_BSE_conduction,
                             nbands=Opt_num_of_bands,
                             mode='BSE',
                             integrate_gamma='sphere', txt=struct+'-6-Log-Optical-BSE.txt')

                # Getting dielectric function spectrum
                parprint("Starting dielectric function calculation...")
                # Writing to files
                bse.get_dielectric_function(filename=struct+'-6-Result-Optical-BSE_dielec.csv',
                                            eta=Opt_eta, w_w=np.linspace(Opt_BSE_min_en, Opt_BSE_max_en, Opt_BSE_num_of_data),
                                            write_eig=struct+'-6-Result-Optical-BSE_eig.dat')
                # Loading dielectric function spectrum to numpy
                dielec = genfromtxt(struct+'-6-Result-Optical-BSE_dielec.csv', delimiter=',')
                # dielec.shape[0] will give us the length of data.
                c_opt = 29979245800
                h_opt = 6.58E-16
                # Initialize arrays
                opt_n_bse = np.array ([1e-6,]*dielec.shape[0])
                opt_k_bse = np.array ([1e-6,]*dielec.shape[0])
                opt_abs_bse = np.array([1e-6,]*dielec.shape[0])
                opt_ref_bse = np.array([1e-6,]*dielec.shape[0])
                # Calculation of other optical data
                for n in range(dielec.shape[0]):
                    opt_n_bse[n] = np.sqrt((np.sqrt(np.square(dielec[n][1])+np.square(dielec[n][2]))+dielec[n][1])/2.0)
                    opt_k_bse[n] = np.sqrt((np.sqrt(np.square(dielec[n][1])+np.square(dielec[n][2]))-dielec[n][1])/2.0)
                    opt_abs_bse[n] = 2*dielec[n][0]*opt_k_bse[n]/(h_opt*c_opt)
                    opt_ref_bse[n] = (np.square(1-opt_n_bse[n])+np.square(opt_k_bse[n]))/(np.square(1+opt_n_bse[n])+np.square(opt_k_bse[n]))
                
                # Saving other data
                with paropen(struct+'-6-Result-Optical-BSE-AllData.dat', 'w') as f1:
                    print("Energy(eV) Eps_real Eps_img Refractive_Index Extinction_Index Absorption(1/cm) Reflectivity", end="\n", file=f1)
                    for n in range(dielec.shape[0]):
                        print(dielec[n][0], dielec[n][1], dielec[n][2], opt_n_bse[n], opt_k_bse[n], opt_abs_bse[n], opt_ref_bse[n], end="\n", file=f1)
                    print (end="\n", file=f1)
                    
                '''
                # DIRECTION IS NOT WORKING FOR A WHILE, IN FUTURE THESE LINES CAN BE USED
                bse.get_dielectric_function(filename=struct+'-6-Result-Optical-BSE_dielec_xdirection.csv',
                                            q_c = [0.0, 0.0, 0.0], direction=0, eta=Opt_eta,
                                            w_w=np.linspace(Opt_BSE_min_en, Opt_BSE_max_en, Opt_BSE_num_of_data),
                                            write_eig=struct+'-6-Result-Optical-BSE_eig_xdirection.dat')
                bse.get_dielectric_function(q_c = [0.0, 0.0, 0.0], direction=1, eta=Opt_eta,
                                            w_w=np.linspace(Opt_BSE_min_en, Opt_BSE_max_en, Opt_BSE_num_of_data),
                                            filename=struct+'-6-Result-Optical-BSE_dielec_ydirection.csv',
                                            write_eig=struct+'-6-Result-Optical-BSE_eig_ydirection.dat')
                bse.get_dielectric_function(q_c = [0.0, 0.0, 0.0], direction=2, eta=Opt_eta,
                                            w_w=np.linspace(Opt_BSE_min_en, Opt_BSE_max_en, Opt_BSE_num_of_data),
                                            filename=struct+'-6-Result-Optical-BSE_dielec_zdirection.csv',
                                            write_eig=struct+'-6-Result-Optical-BSE_eig_zdirection.dat')

                # Loading dielectric function spectrum to numpy
                dielec_x = genfromtxt(struct+'-6-Result-Optical-BSE_dielec_xdirection.csv', delimiter=',')
                dielec_y = genfromtxt(struct+'-6-Result-Optical-BSE_dielec_ydirection.csv', delimiter=',')
                dielec_z = genfromtxt(struct+'-6-Result-Optical-BSE_dielec_zdirection.csv', delimiter=',')
                # dielec_x.shape[0] will give us the length of data.
                # c and h
                c_opt = 29979245800
                h_opt = 6.58E-16
                #c_opt = 1
                #h_opt = 1

                # Initialize arrays
                opt_n_bse_x = np.array ([1e-6,]*dielec_x.shape[0])
                opt_k_bse_x = np.array ([1e-6,]*dielec_x.shape[0])
                opt_abs_bse_x = np.array([1e-6,]*dielec_x.shape[0])
                opt_ref_bse_x = np.array([1e-6,]*dielec_x.shape[0])
                opt_n_bse_y = np.array ([1e-6,]*dielec_y.shape[0])
                opt_k_bse_y = np.array ([1e-6,]*dielec_y.shape[0])
                opt_abs_bse_y = np.array([1e-6,]*dielec_y.shape[0])
                opt_ref_bse_y = np.array([1e-6,]*dielec_y.shape[0])
                opt_n_bse_z = np.array ([1e-6,]*dielec_z.shape[0])
                opt_k_bse_z = np.array ([1e-6,]*dielec_z.shape[0])
                opt_abs_bse_z = np.array([1e-6,]*dielec_z.shape[0])
                opt_ref_bse_z = np.array([1e-6,]*dielec_z.shape[0])

                # Calculation of other optical data
                for n in range(dielec_x.shape[0]):
                    # x-direction
                    opt_n_bse_x[n] = np.sqrt((np.sqrt(np.square(dielec_x[n][1])+np.square(dielec_x[n][2]))+dielec_x[n][1])/2.0)
                    opt_k_bse_x[n] = np.sqrt((np.sqrt(np.square(dielec_x[n][1])+np.square(dielec_x[n][2]))-dielec_x[n][1])/2.0)
                    opt_abs_bse_x[n] = 2*dielec_x[n][0]*opt_k_bse_x[n]/(h_opt*c_opt)
                    opt_ref_bse_x[n] = (np.square(1-opt_n_bse_x[n])+np.square(opt_k_bse_x[n]))/(np.square(1+opt_n_bse_x[n])+np.square(opt_k_bse_x[n]))
                    # y-direction
                    opt_n_bse_y[n] = np.sqrt((np.sqrt(np.square(dielec_y[n][1])+np.square(dielec_y[n][2]))+dielec_y[n][1])/2.0)
                    opt_k_bse_y[n] = np.sqrt((np.sqrt(np.square(dielec_y[n][1])+np.square(dielec_y[n][2]))-dielec_y[n][1])/2.0)
                    opt_abs_bse_y[n] = 2*dielec_y[n][0]*opt_k_bse_y[n]/(h_opt*c_opt)
                    opt_ref_bse_y[n] = (np.square(1-opt_n_bse_y[n])+np.square(opt_k_bse_y[n]))/(np.square(1+opt_n_bse_y[n])+np.square(opt_k_bse_y[n]))
                    # z-direction
                    opt_n_bse_z[n] = np.sqrt((np.sqrt(np.square(dielec_z[n][1])+np.square(dielec_z[n][2]))+dielec_z[n][1])/2.0)
                    opt_k_bse_z[n] = np.sqrt((np.sqrt(np.square(dielec_z[n][1])+np.square(dielec_z[n][2]))-dielec_z[n][1])/2.0)
                    opt_abs_bse_z[n] = 2*dielec_z[n][0]*opt_k_bse_z[n]/(h_opt*c_opt)
                    opt_ref_bse_z[n] = (np.square(1-opt_n_bse_z[n])+np.square(opt_k_bse_z[n]))/(np.square(1+opt_n_bse_z[n])+np.square(opt_k_bse_z[n]))

                # Saving other data for x-direction
                with paropen(struct+'-6-Result-Optical-BSE-AllData_xdirection.dat', 'w') as f1:
                    print("Energy(eV) Eps_real Eps_img Refractive_Index Extinction_Index Absorption(1/cm) Reflectivity", end="\n", file=f1)
                    for n in range(dielec_x.shape[0]):
                        print(dielec_x[n][0], dielec_x[n][1], dielec_x[n][2], opt_n_bse_x[n], opt_k_bse_x[n], opt_abs_bse_x[n], opt_ref_bse_x[n], end="\n", file=f1)
                    print (end="\n", file=f1)

                # Saving other data for y-direction
                with paropen(struct+'-6-Result-Optical-BSE-AllData_ydirection.dat', 'w') as f1:
                    print("Energy(eV) Eps_real Eps_img Refractive_Index Extinction_Index Absorption(1/cm) Reflectivity", end="\n", file=f1)
                    for n in range(dielec_y.shape[0]):
                        print(dielec_y[n][0], dielec_y[n][1], dielec_y[n][2], opt_n_bse_y[n], opt_k_bse_y[n], opt_abs_bse_y[n], opt_ref_bse_y[n], end="\n", file=f1)
                    print (end="\n", file=f1)

                # Saving other data for z-direction
                with paropen(struct+'-6-Result-Optical-BSE-AllData_zdirection.dat', 'w') as f1:
                    print("Energy(eV) Eps_real Eps_img Refractive_Index Extinction_Index Absorption(1/cm) Reflectivity", end="\n", file=f1)
                    for n in range(dielec_z.shape[0]):
                        print(dielec_z[n][0], dielec_z[n][1], dielec_z[n][2], opt_n_bse_z[n], opt_k_bse_z[n], opt_abs_bse_z[n], opt_ref_bse_z[n], end="\n", file=f1)
                    print (end="\n", file=f1)
               '''
            
            elif Opt_calc_type == 'RPA':
                parprint('Starting RPA calculations')
                df = DielectricFunction(calc=struct+'-6-Result-Optical.gpw',
                                        frequencies={'type': 'nonlinear', 'domega0': Opt_domega0, 'omega2': Opt_omega2},
                                        eta=Opt_eta, intraband=False, hilbert=False,
                                        ecut=Opt_cut_of_energy, txt=struct+'-6-Log-Optical-RPA.txt')
                # Writing to files as: omega, nlfc.real, nlfc.imag, lfc.real, lfc.imag
                # Here lfc is local field correction
                # Getting dielectric function spectrum
                parprint("Starting dielectric function calculation...")
                df.get_dielectric_function(direction='x', filename=struct+'-6-Result-Optical-RPA_dielec_xdirection.csv')
                df.get_dielectric_function(direction='y', filename=struct+'-6-Result-Optical-RPA_dielec_ydirection.csv')
                df.get_dielectric_function(direction='z', filename=struct+'-6-Result-Optical-RPA_dielec_zdirection.csv')

                # Loading dielectric function spectrum to numpy
                dielec_x = genfromtxt(struct+'-6-Result-Optical-RPA_dielec_xdirection.csv', delimiter=',')
                dielec_y = genfromtxt(struct+'-6-Result-Optical-RPA_dielec_ydirection.csv', delimiter=',')
                dielec_z = genfromtxt(struct+'-6-Result-Optical-RPA_dielec_zdirection.csv', delimiter=',')
                # dielec_x.shape[0] will give us the length of data.
                # c and h
                c_opt = 29979245800
                h_opt = 6.58E-16
                #c_opt = 1
                #h_opt = 1
                # ---- NLFC ----
                # Initialize arrays for NLFC
                opt_n_nlfc_x = np.array ([1e-6,]*dielec_x.shape[0])
                opt_k_nlfc_x = np.array ([1e-6,]*dielec_x.shape[0])
                opt_abs_nlfc_x = np.array([1e-6,]*dielec_x.shape[0])
                opt_ref_nlfc_x = np.array([1e-6,]*dielec_x.shape[0])
                opt_n_nlfc_y = np.array ([1e-6,]*dielec_y.shape[0])
                opt_k_nlfc_y = np.array ([1e-6,]*dielec_y.shape[0])
                opt_abs_nlfc_y = np.array([1e-6,]*dielec_y.shape[0])
                opt_ref_nlfc_y = np.array([1e-6,]*dielec_y.shape[0])
                opt_n_nlfc_z = np.array ([1e-6,]*dielec_z.shape[0])
                opt_k_nlfc_z = np.array ([1e-6,]*dielec_z.shape[0])
                opt_abs_nlfc_z = np.array([1e-6,]*dielec_z.shape[0])
                opt_ref_nlfc_z = np.array([1e-6,]*dielec_z.shape[0])

                # Calculation of other optical spectrum for NLFC
                for n in range(dielec_x.shape[0]):
                    # NLFC-x
                    opt_n_nlfc_x[n] = np.sqrt((np.sqrt(np.square(dielec_x[n][1])+np.square(dielec_x[n][2]))+dielec_x[n][1])/2.0)
                    opt_k_nlfc_x[n] = np.sqrt((np.sqrt(np.square(dielec_x[n][1])+np.square(dielec_x[n][2]))-dielec_x[n][1])/2.0)
                    opt_abs_nlfc_x[n] = 2*dielec_x[n][0]*opt_k_nlfc_x[n]/(h_opt*c_opt)
                    opt_ref_nlfc_x[n] = (np.square(1-opt_n_nlfc_x[n])+np.square(opt_k_nlfc_x[n]))/(np.square(1+opt_n_nlfc_x[n])+np.square(opt_k_nlfc_x[n]))
                    # NLFC-y
                    opt_n_nlfc_y[n] = np.sqrt((np.sqrt(np.square(dielec_y[n][1])+np.square(dielec_y[n][2]))+dielec_y[n][1])/2.0)
                    opt_k_nlfc_y[n] = np.sqrt((np.sqrt(np.square(dielec_y[n][1])+np.square(dielec_y[n][2]))-dielec_y[n][1])/2.0)
                    opt_abs_nlfc_y[n] = 2*dielec_y[n][0]*opt_k_nlfc_y[n]/(h_opt*c_opt)
                    opt_ref_nlfc_y[n] = (np.square(1-opt_n_nlfc_y[n])+np.square(opt_k_nlfc_y[n]))/(np.square(1+opt_n_nlfc_y[n])+np.square(opt_k_nlfc_y[n]))
                    # NLFC-z
                    opt_n_nlfc_z[n] = np.sqrt((np.sqrt(np.square(dielec_z[n][1])+np.square(dielec_z[n][2]))+dielec_z[n][1])/2.0)
                    opt_k_nlfc_z[n] = np.sqrt((np.sqrt(np.square(dielec_z[n][1])+np.square(dielec_z[n][2]))-dielec_z[n][1])/2.0)
                    opt_abs_nlfc_z[n] = 2*dielec_z[n][0]*opt_k_nlfc_z[n]/(h_opt*c_opt)
                    opt_ref_nlfc_z[n] = (np.square(1-opt_n_nlfc_z[n])+np.square(opt_k_nlfc_z[n]))/(np.square(1+opt_n_nlfc_z[n])+np.square(opt_k_nlfc_z[n]))

                # Saving NLFC other optical spectrum for x-direction
                with paropen(struct+'-6-Result-Optical-RPA-NLFC-AllData_xdirection.dat', 'w') as f1:
                    print("Energy(eV) Eps_real Eps_img Refractive_Index Extinction_Index Absorption(1/cm) Reflectivity", end="\n", file=f1)
                    for n in range(dielec_x.shape[0]):
                        print(dielec_x[n][0], dielec_x[n][1], dielec_x[n][2], opt_n_nlfc_x[n], opt_k_nlfc_x[n], opt_abs_nlfc_x[n], opt_ref_nlfc_x[n], end="\n", file=f1)
                    print (end="\n", file=f1)

                # Saving NLFC other optical spectrum for y-direction
                with paropen(struct+'-6-Result-Optical-RPA-NLFC-AllData_ydirection.dat', 'w') as f1:
                    print("Energy(eV) Eps_real Eps_img Refractive_Index Extinction_Index Absorption(1/cm) Reflectivity", end="\n", file=f1)
                    for n in range(dielec_y.shape[0]):
                        print(dielec_y[n][0], dielec_y[n][1], dielec_y[n][2], opt_n_nlfc_y[n], opt_k_nlfc_y[n], opt_abs_nlfc_y[n], opt_ref_nlfc_y[n], end="\n", file=f1)
                    print (end="\n", file=f1)

                # Saving NLFC other optical spectrum for z-direction
                with paropen(struct+'-6-Result-Optical-RPA-NLFC-AllData_zdirection.dat', 'w') as f1:
                    print("Energy(eV) Eps_real Eps_img Refractive_Index Extinction_Index Absorption(1/cm) Reflectivity", end="\n", file=f1)
                    for n in range(dielec_z.shape[0]):
                        print(dielec_z[n][0], dielec_z[n][1], dielec_z[n][2], opt_n_nlfc_z[n], opt_k_nlfc_z[n], opt_abs_nlfc_z[n], opt_ref_nlfc_z[n], end="\n", file=f1)
                    print (end="\n", file=f1)

                # ---- LFC ----
                # Initialize arrays for LFC
                opt_n_lfc_x = np.array ([1e-6,]*dielec_x.shape[0])
                opt_k_lfc_x = np.array ([1e-6,]*dielec_x.shape[0])
                opt_abs_lfc_x = np.array([1e-6,]*dielec_x.shape[0])
                opt_ref_lfc_x = np.array([1e-6,]*dielec_x.shape[0])
                opt_n_lfc_y = np.array ([1e-6,]*dielec_y.shape[0])
                opt_k_lfc_y = np.array ([1e-6,]*dielec_y.shape[0])
                opt_abs_lfc_y = np.array([1e-6,]*dielec_y.shape[0])
                opt_ref_lfc_y = np.array([1e-6,]*dielec_y.shape[0])
                opt_n_lfc_z = np.array ([1e-6,]*dielec_z.shape[0])
                opt_k_lfc_z = np.array ([1e-6,]*dielec_z.shape[0])
                opt_abs_lfc_z = np.array([1e-6,]*dielec_z.shape[0])
                opt_ref_lfc_z = np.array([1e-6,]*dielec_z.shape[0])

                # Calculation of other optical spectrum for LFC
                for n in range(dielec_x.shape[0]):
                    # LFC-x
                    opt_n_lfc_x[n] = np.sqrt((np.sqrt(np.square(dielec_x[n][3])+np.square(dielec_x[n][4]))+dielec_x[n][3])/2.0)
                    opt_k_lfc_x[n] = np.sqrt((np.sqrt(np.square(dielec_x[n][3])+np.square(dielec_x[n][4]))-dielec_x[n][3])/2.0)
                    opt_abs_lfc_x[n] = 2*dielec_x[n][0]*opt_k_nlfc_x[n]/(h_opt*c_opt)
                    opt_ref_lfc_x[n] = (np.square(1-opt_n_lfc_x[n])+np.square(opt_k_lfc_x[n]))/(np.square(1+opt_n_lfc_x[n])+np.square(opt_k_lfc_x[n]))
                    # LFC-y
                    opt_n_lfc_y[n] = np.sqrt((np.sqrt(np.square(dielec_y[n][3])+np.square(dielec_y[n][4]))+dielec_y[n][3])/2.0)
                    opt_k_lfc_y[n] = np.sqrt((np.sqrt(np.square(dielec_y[n][3])+np.square(dielec_y[n][4]))-dielec_y[n][3])/2.0)
                    opt_abs_lfc_y[n] = 2*dielec_y[n][0]*opt_k_lfc_y[n]/(h_opt*c_opt)
                    opt_ref_lfc_y[n] = (np.square(1-opt_n_lfc_y[n])+np.square(opt_k_lfc_y[n]))/(np.square(1+opt_n_lfc_y[n])+np.square(opt_k_lfc_y[n]))
                    # LFC-z
                    opt_n_lfc_z[n] = np.sqrt((np.sqrt(np.square(dielec_z[n][3])+np.square(dielec_z[n][4]))+dielec_z[n][3])/2.0)
                    opt_k_lfc_z[n] = np.sqrt((np.sqrt(np.square(dielec_z[n][3])+np.square(dielec_z[n][4]))-dielec_z[n][3])/2.0)
                    opt_abs_lfc_z[n] = 2*dielec_z[n][0]*opt_k_lfc_z[n]/(h_opt*c_opt)
                    opt_ref_lfc_z[n] = (np.square(1-opt_n_lfc_z[n])+np.square(opt_k_lfc_z[n]))/(np.square(1+opt_n_lfc_z[n])+np.square(opt_k_lfc_z[n]))

                # Saving LFC other optical spectrum for x-direction
                with paropen(struct+'-6-Result-Optical-RPA-LFC-AllData_xdirection.dat', 'w') as f1:
                    print("Energy(eV) Eps_real Eps_img Refractive_Index Extinction_Index Absorption(1/cm) Reflectivity", end="\n", file=f1)
                    for n in range(dielec_x.shape[0]):
                        print(dielec_x[n][0], dielec_x[n][3], dielec_x[n][4], opt_n_lfc_x[n], opt_k_lfc_x[n], opt_abs_lfc_x[n], opt_ref_lfc_x[n], end="\n", file=f1)
                    print (end="\n", file=f1)

                # Saving LFC other optical spectrum for y-direction
                with paropen(struct+'-6-Result-Optical-RPA-LFC-AllData_ydirection.dat', 'w') as f1:
                    print("Energy(eV) Eps_real Eps_img Refractive_Index Extinction_Index Absorption(1/cm) Reflectivity", end="\n", file=f1)
                    for n in range(dielec_y.shape[0]):
                        print(dielec_y[n][0], dielec_y[n][3], dielec_y[n][4], opt_n_lfc_y[n], opt_k_lfc_y[n], opt_abs_lfc_y[n], opt_ref_lfc_y[n], end="\n", file=f1)
                    print (end="\n", file=f1)

                # Saving LFC other optical spectrum for z-direction
                with paropen(struct+'-6-Result-Optical-RPA-LFC-AllData_zdirection.dat', 'w') as f1:
                    print("Energy(eV) Eps_real Eps_img Refractive_Index Extinction_Index Absorption(1/cm) Reflectivity", end="\n", file=f1)
                    for n in range(dielec_z.shape[0]):
                        print(dielec_z[n][0], dielec_z[n][3], dielec_z[n][4], opt_n_lfc_z[n], opt_k_lfc_z[n], opt_abs_lfc_z[n], opt_ref_lfc_z[n], end="\n", file=f1)
                    print (end="\n", file=f1)

            else:
                parprint('\033[91mERROR:\033[0mUnknown optical calculation type.')
                quit()

        elif Mode == 'LCAO':
            parprint('\033[91mERROR:\033[0mNot implemented in LCAO mode yet.')
        else:
            parprint('\033[91mERROR:\033[0mNot implemented in FD mode yet.')
        # Finish Optical calc
        time62 = time.time()
        # Write timings of calculation
        with paropen(struct+'-7-Result-Log-Timings.txt', 'a') as f1:
            print('Optical calculation: ', round((time62-time61),2), end="\n", file=f1)

# Elastic related functions
from ase.spacegroup import get_spacegroup
import numpy as np

def reconstruct_full_tensor(independent, atoms):
    """
    Reconstruct the full 6x6 elastic tensor from the independent elastic constants
    returned by get_elastic_tensor(), based on the spacegroup determined by ASE.
    
    Parameters:
      independent : NumPy array of independent elastic constants.
      atoms       : ASE Atoms object.
      
    Returns:
      A 6x6 NumPy array representing the full elastic tensor.
      
    Implemented cases:
      - 3 independent constants: assumed cubic or isotropic, 
        mapped as:
            C11  C12  C12   0    0    0
            C12  C11  C12   0    0    0
            C12  C12  C11   0    0    0
             0    0    0   C44   0    0
             0    0    0    0   C44   0
             0    0    0    0    0   (C11-C12)/2
      - 5 independent constants: assumed hexagonal, mapped as:
            C11  C12  C13   0    0    0
            C12  C11  C13   0    0    0
            C13  C13  C33   0    0    0
             0    0    0   C44   0    0
             0    0    0    0   C44   0
             0    0    0    0    0   (C11-C12)/2
      - 21 independent constants: assumed triclinic; these are
        taken as the lower triangular elements (in order) and then mirrored.
    """
    sg = get_spacegroup(atoms, symprec=1e-2)
    symbol = sg.symbol
    parprint(f"[DEBUG] Reconstructing tensor from {independent.size} independent constants; spacegroup: {symbol}")
    if independent.size == 3:
        # Assume cubic or isotropic system.
        C11, C12, C44 = independent
        full = np.array([
            [C11, C12, C12, 0,   0,   0],
            [C12, C11, C12, 0,   0,   0],
            [C12, C12, C11, 0,   0,   0],
            [0,   0,   0,   C44, 0,   0],
            [0,   0,   0,   0,   C44, 0],
            [0,   0,   0,   0,   0,   (C11-C12)/2]
        ])
        return full
    elif independent.size == 5:
        # Assume hexagonal symmetry.
        C11, C12, C13, C33, C44 = independent
        C66 = (C11 - C12) / 2
        full = np.array([
            [C11, C12, C13, 0,   0,   0],
            [C12, C11, C13, 0,   0,   0],
            [C13, C13, C33, 0,   0,   0],
            [0,   0,   0,   C44, 0,   0],
            [0,   0,   0,   0,   C44, 0],
            [0,   0,   0,   0,   0,   C66]
        ])
        return full
    elif independent.size == 21:
        full = np.zeros((6,6))
        idx = 0
        for i in range(6):
            for j in range(i+1):
                full[i,j] = independent[idx]
                idx += 1
        # Mirror the lower triangle to the upper triangle.
        full = full + full.T - np.diag(np.diag(full))
        return full
    else:
        raise ValueError("Reconstruction for symmetry with {} independent constants is not implemented.".format(independent.size))


# Phonon related functions
# The remaining functions related to phonon calculations in this file are MIT-licensed by (C) 2020 Michael Lamparski

def get_band_path(atoms, path_str, npoints, path_frac=None, labels=None):
    from ase.dft.kpoints import bandpath

    atoms = convert_atoms_to_ase(atoms)
    if path_str is None:
        path_str = atoms.get_cell().bandpath().path

    # Commas are part of ase's supported syntax, but we'll take care of them
    # ourselves to make it easier to get things the way phonopy wants them
    if path_frac is None:
        path_frac = []
        for substr in path_str.split(','):
            path = bandpath(substr, atoms.get_cell()[...], npoints=1)
            path_frac.append(path.kpts)

    if labels is None:
        labels = []
        for substr in path_str.split(','):
            path = bandpath(substr, atoms.get_cell()[...], npoints=1)

            _, _, substr_labels = path.get_linear_kpoint_axis()
            labels.extend(['$\\Gamma$' if s == 'G' else s for s in substr_labels])

    qpoints, connections = get_band_qpoints_and_path_connections(path_frac, npoints=npoints)
    return qpoints, labels, connections

def run_gpaw_all(calc, phonon):
    return [ run_gpaw(calc, supercell) for supercell in phonon.supercells_with_displacements ]

def run_gpaw(calc, cell):
    cell = convert_atoms_to_ase(cell)
    cell.set_calculator(calc)
    forces = cell.get_forces()
    drift_force = forces.sum(axis=0)
    with paropen(struct+'-5-Log-Phonon-Phonopy.txt', 'a') as f2:
        print(("[Phonopy] Drift force:" + "%11.5f" * 3) % tuple(drift_force), end="\n", file=f2)
    # Simple translational invariance
    for force in forces:
        force -= drift_force / forces.shape[0]
    return forces

#--------------------------------------------------------------------


def load_or_compute_force(path, calc, atoms):
    if os.path.exists(path):
        with paropen(struct+'-5-Log-Phonon-Phonopy.txt', 'a') as f2:
            print('Reading {!r}'.format(path), end="\n", file=f2)
        return np.load(path)

    else:
        with paropen(struct+'-5-Log-Phonon-Phonopy.txt', 'a') as f2:
            print('Computing {!r}'.format(path), end="\n", file=f2)
        force_set = run_gpaw(calc, atoms)
        np.save(path, force_set)
        return force_set

#--------------------------------------------------------------------

def convert_atoms_to_ase(atoms):
    return Atoms(
        symbols=atoms.get_chemical_symbols(),
        scaled_positions=atoms.get_scaled_positions(),
        cell=atoms.get_cell(),
        pbc=True
    )

def convert_atoms_to_phonopy(atoms):
    return PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        scaled_positions=atoms.get_scaled_positions(),
        cell=atoms.get_cell()
    )


# End of phonon related functions------------------------------

# Projected Band Structure related functions-------------------

def projected_weights(calc):
    ns = calc.get_number_of_spins()
    atom_num = calc.atoms.get_atomic_numbers()
    atoms = calc.atoms
    
    # Defining the atoms and angular momentum to project onto.
    lan = range(58, 72)
    act = range(90, 104)
    atom_num = np.asarray(atom_num)
    ang_mom_a = {}
    atoms = Atoms(numbers=atom_num)
    magnetic_elements = {'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                'Y', 'Zr', 'Nb', 'Mo', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In',
                'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl'}

    for a, (z, magn) in enumerate(zip(atom_num, magnetic_elements)):
        if z in lan or z in act:
            ang_mom_a[a] = 'spdf'
        else:
            ang_mom_a[a] = 'spd' if magn else 'sp'

    # For each unique atom
    a_x = [a for a in ang_mom_a for _ in ang_mom_a[a]]
    ang_mom_x = [ang_mom for a in ang_mom_a for ang_mom in ang_mom_a[a]]

    # Get i index for each unique symbol
    sym_ang_mom_i = []
    i_x = []
    for a, ang_mom in zip(a_x, ang_mom_x):
        symbol = atoms.symbols[a]
        sym_ang_mom = '.'.join([str(symbol), str(ang_mom)])
        if sym_ang_mom in sym_ang_mom_i:
            i = sym_ang_mom_i.index(sym_ang_mom)
        else:
            i = len(sym_ang_mom_i)
            sym_ang_mom_i.append(sym_ang_mom)
        i_x.append(i)

    nk, nb = len(calc.get_ibz_k_points()), calc.get_number_of_bands()
    projector_weight_skni = np.zeros((ns, nk, nb, len(sym_ang_mom_i)))
    ali_x = [(a, ang_mom, i) for (a, ang_mom, i) in zip(a_x, ang_mom_x, i_x)]
    
    for _, (a, ang_mom, i) in enumerate(ali_x):
        # Extract weights
        for s in range(ns):
            __, weights = raw_orbital_LDOS(calc, a, s, ang_mom)
            projector_weight_kn = weights.reshape((nk, nb))
            projector_weight_kn /= calc.wfs.kd.weight_k[:, np.newaxis]
            projector_weight_skni[s, :, :, i] += projector_weight_kn

    return projector_weight_skni, sym_ang_mom_i

# End of Projected Band Structure related functions----------------

if __name__ == "__main__":
    #
    # DEFAULT VALUES
    # These values (with bulk configuration) can be used to run this script without using inputfile (py file)
    # and configuration file (cif file).
    # -------------------------------------------------------------
    Mode = 'PW'             # Use PW, LCAO, FD  (PW is more accurate, LCAO is quicker mostly.)
    # -------------------------------------------------------------
    Ground_calc = False     # Ground state calculations
    Geo_optim = False       # Geometric optimization with LFBGS
    Elastic_calc = False    # Elastic calculation
    DOS_calc = False         # DOS calculation
    Band_calc = False        # Band structure calculation
    Density_calc = False    # Calculate the all-electron density?
    Phonon_calc = False     # Phonon calculations
    Optical_calc = False     # Calculate the optical properties

    # -------------------------------------------------------------
    # Parameters
    # -------------------------------------------------------------
    # GEOMETRY
    Optimizer = 'QuasiNewton' # QuasiNewton, GPMin, LBFGS or FIRE
    Max_F_tolerance = 0.05 	# Maximum force tolerance in LBFGS geometry optimization. Unit is eV/Ang.
    Max_step = 0.1          # How far is a single atom allowed to move. Default is 0.2 Ang.
    Alpha = 60.0            # LBFGS only: Initial guess for the Hessian (curvature of energy surface)
    Damping = 1.0           # LBFGS only: The calculated step is multiplied with this number before added to the positions
    Fix_symmetry = False    # True for preserving the spacegroup symmetry during optimisation
    # Which components of strain will be relaxed: EpsX, EpsY, EpsZ, ShearYZ, ShearXZ, ShearXY
    # Example: For a x-y 2D nanosheet only first 2 component will be true
    Relax_cell = [False, False, False, False, False, False]
    Hydrostatic_pressure = 0.0 # GPa

    # GROUND ----------------------
    Cut_off_energy = 340 	# eV
    Ground_kpts_density = None     # pts per Å^-1  If the user prefers to use this, Ground_kpts_x,y,z will not be used automatically.
    Ground_kpts_x = 5 	# kpoints in x direction
    Ground_kpts_y = 5	# kpoints in y direction
    Ground_kpts_z = 5	# kpoints in z direction
    Ground_gpts_density = None     # (for LCAO) Unit is Å. If the user prefers to use this, Ground_gpts_x,y,z will not be used automatically.
    Ground_gpts_x = 8              # grid points in x direction (for LCAO)
    Ground_gpts_y = 8              # grid points in y direction (for LCAO)
    Ground_gpts_z = 8              # grid points in z direction (for LCAO)
    Setup_params = {}            # Can be used like {'N': ':p,6.0'} for Hubbard, can also be used for many corrections.https://wiki.fysik.dtu.dk/gpaw/devel/setups.html#gpaw.setup.Setup For none use {}
    XC_calc = 'LDA'         # Exchange-Correlation, choose one: LDA, PBE, GLLBSCM, HSE06, HSE03, revPBE, RPBE, PBE0, EXX, B3LYP
    Ground_convergence = {}   # Convergence items for ground state calculations
    Occupation = {'name': 'fermi-dirac', 'width': 0.05}  # Refer to GPAW docs: https://wiki.fysik.dtu.dk/gpaw/documentation/basic.html#occupation-numbers
    Mixer_type = MixerSum(0.1, 3, 50) # MixerSum(beta,nmaxold, weight) default:(0.1,3,50), you can try (0.02, 5, 100) and (0.05, 5, 50)
    Spin_calc = False        # Spin polarized calculation?
    Magmom_per_atom = 1.0    # Magnetic moment per atom
    Magmom_single_atom = None # Magnetic moment for a single atom [atom_no, magmom]
    Total_charge = 0.0       # Total charge. Normally 0.0 for a neutral system.

    # DOS ----------------------
    DOS_npoints = 501                # Number of points
    DOS_width = 0.1                  # Width of Gaussian smearing. Use 0.0 for linear tetrahedron interpolation
    DOS_convergence = {}             # Convergence items for DOS calculations

    # BAND ----------------------
    Gamma = True
    Band_path = 'LGL'	    # Brillouin zone high symmetry points
    Band_npoints = 61		# Number of points between high symmetry points
    Energy_max = 5 		# eV. It is the maximum energy value for band structure and DOS figures.
    Energy_min = -5     # eV. It is the minimum energy value for band structure and DOS figures.
    Band_convergence = {'bands':8}   # Convergence items for band calculations

    # ELECTRON DENSITY ----------------------
    Refine_grid = 4             # refine grid for all electron density (1, 2 [=default] and 4)

    # PHONON -------------------------
    Phonon_PW_cutoff = 400
    Phonon_kpts_x = 3
    Phonon_kpts_y = 3
    Phonon_kpts_z = 3
    Phonon_supercell = np.diag([2, 2, 2])
    Phonon_displacement = 1e-3
    Phonon_path = 'LGL'	    # Brillouin zone high symmetry points
    Phonon_npoints = 61		# Number of points between high symmetry points
    Phonon_acoustic_sum_rule = True

    # OPTICAL ----------------------
    Opt_calc_type = 'BSE'         # BSE or RPA
    Opt_shift_en = 0.0          # Shifting of the energy
    Opt_BSE_valence = range(0,3)  # Valence bands that will be used in BSE calculation
    Opt_BSE_conduction = range(4,7) # Conduction bands that will be used in BSE calculation
    Opt_BSE_min_en = 0.0       # Results will be started from this energy (BSE only)
    Opt_BSE_max_en = 20.0      # Results will be ended at this energy (BSE only)
    Opt_BSE_num_of_data = 1001   # Number of data points in BSE  calculation
    Opt_num_of_bands = 8	# Number of bands
    Opt_FD_smearing = 0.05       # Fermi Dirac smearing for optical calculations
    Opt_eta = 0.05             # Eta for Optical calculations
    Opt_domega0 = 0.05         # Domega0 for Optical calculations
    Opt_omega2 = 5.0           # Frequency at which the non-lin freq grid has doubled the spacing
    Opt_cut_of_energy = 100             # Cut-off energy for optical calculations
    Opt_nblocks = world.size            # Split matrices in nblocks blocks and distribute them G-vectors
                            # or frequencies over processes. Also can use world.size

    #GENERAL ----------------------
    MPI_cores = 4            # This is for gg.py. Not used in this script.
    Localisation = "en_UK"

    # -------------------------------------------------------------
    # Default Bulk Configuration
    # -------------------------------------------------------------
    bulk_configuration = Atoms(
        [
        Atom('C', ( 0.0, 0.0, 5.0 )),
        Atom('C', ( -1.2339999999999995, 2.1373506965399947, 5.0 )),
        Atom('C', ( 2.4679999999999995, 0.0, 5.0 )),
        Atom('C', ( 1.234, 2.1373506965399947, 5.0 )),
        Atom('C', ( 2.468000000230841e-06, 1.424899039459532, 5.0 )),
        Atom('C', ( -1.2339975319999992, 3.5622497359995267, 5.0 )),
        Atom('C', ( 2.4680024680000003, 1.424899039459532, 5.0 )),
        Atom('C', ( 1.234002468000001, 3.5622497359995267, 5.0 )),
        ],
        cell=[(4.936, 0.0, 0.0), (-2.467999999999999, 4.274701393079989, 0.0), (0.0, 0.0, 20.0)],
        pbc=True,
        )
    # ------------------ Localisation Tables. You can add your language below --------------------------
    dos_xlabel = dict(en_UK='', tr_TR='', de_DE='', fr_FR='', ru_RU='', zh_CN='', ko_KR='', ja_JP='')
    dos_ylabel = dict(en_UK='', tr_TR='', de_DE='', fr_FR='', ru_RU='', zh_CN='', ko_KR='', ja_JP='')
    band_ylabel = dict(en_UK='', tr_TR='', de_DE='', fr_FR='', ru_RU='', zh_CN='', ko_KR='', ja_JP='')
    
    # ENGLISH (en_UK) - by S.B. Lisesivdin
    ## Figures
    dos_xlabel["en_UK"]='Energy [eV]'
    dos_ylabel["en_UK"]='DOS [1/eV]'
    band_ylabel["en_UK"]='Energy [eV]'

    # TURKISH (tr_TR) - by S.B. Lisesivdin
    ## Figures
    dos_xlabel["tr_TR"]='Enerji [eV]'
    dos_ylabel["tr_TR"]='Durum Yoğunluğu [1/eV]'
    band_ylabel["tr_TR"]='Enerji [eV]'
    
    # GERMAN (de_DE) - created with AI
    ## Figures
    dos_xlabel["de_DE"]='Energie [eV]'
    dos_ylabel["de_DE"]='Zustandsdichte [1/eV]'
    band_ylabel["de_DE"]='Energie [eV]'

    # FRENCH (fr_FR) - created with AI
    ## Figures
    dos_xlabel["fr_FR"]='Énergie [eV]'
    dos_ylabel["fr_FR"]='DOS [1/eV]'
    band_ylabel["fr_FR"]='Énergie [eV]'

    # RUSSIAN (ru_RU) - created with AI
    ## Figures
    dos_xlabel["ru_RU"]='Энергия [эВ]'
    dos_ylabel["ru_RU"]='Плотность состояний [1/эВ]'
    band_ylabel["ru_RU"]='Энергия [эВ]'

    # CHINESE (zh_CN) - created with AI
    ## Figures
    dos_xlabel["zh_CN"]='能量 [eV]'
    dos_ylabel["zh_CN"]='态密度 [1/eV]'
    band_ylabel["zh_CN"]='能量 [eV]'

    # KOREAN (ko_KR) - created with AI
    ## Figures
    dos_xlabel["ko_KR"]='에너지 [eV]'
    dos_ylabel["ko_KR"]='상태 밀도 [1/eV]'
    band_ylabel["ko_KR"]='에너지 [eV]'

    # JAPANESE (ja_JP) - created with AI
    ## Figures
    dos_xlabel["ja_JP"]='エネルギー [eV]'
    dos_ylabel["ja_JP"]='状態密度 [1/eV]'
    band_ylabel["ja_JP"]='エネルギー [eV]'
        
    # ------------------ End of Localisation Tables --------------------------
    
    # Version
    __version__ = "v25.10.1b1"

    parser = ArgumentParser(prog ='gpawtools.py', description=Description, formatter_class=RawFormatter)
    parser.add_argument("-i", "--input", dest = "inputfile", help="Use input file for calculation variables (also you can insert geometry)")
    parser.add_argument("-g", "--geometry",dest ="geometryfile", help="Use CIF file for geometry")
    parser.add_argument("-v", "--version", dest="version", action='store_true')
    parser.add_argument("-e", "--energy", dest="energymeas", action='store_true')
    parser.add_argument("-d", "--drawfigures", dest="drawfigs", action='store_true', help="Draws DOS and band structure figures at the end of calculation.")

    args = None

    # Parse arguments
    try:
        if world.rank == 0:
            args = parser.parse_args()
    finally:
        args = broadcast(args, root=0, comm=world)

    if args is None:
        parprint("No arguments used.")
        quit()

    # DEFAULT VALUES
    energymeas = False
    inFile = None
    drawfigs = False
    configpath = None
    Outdirname = ''

    try:
        if args.version == True:
            import gpaw
            import ase
            import phonopy
            try:
                response = requests.get("https://api.github.com/repos/sblisesivdin/pint/releases/latest", timeout=5)
                parprint('-----------------------------------------------------------------------------')
                parprint('\033[95mpint:\033[0m Version information: '+str(__version__))
                parprint('  uses GPAW '+gpaw.__version__+', ASE '+ase.__version__+' and PHONOPY '+phonopy.__version__)
                parprint('-----------------------------------------------------------------------------')
                parprint('The latest STABLE release is '+response.json()["tag_name"]+',')
                parprint('which is released at '+response.json()["published_at"]+'.')
                parprint('-----------------------------------------------------------------------------')
                parprint('You can download the latest STABLE tarball, zipball or DEVELOPMENT zipball:')
                parprint(response.json()["tarball_url"])
                parprint(response.json()["zipball_url"])
                parprint('https://github.com/sblisesivdin/pint/archive/refs/heads/main.zip')
            except (requests.ConnectionError, requests.Timeout) as exception:
                parprint('-----------------------------------------------------------------------------')
                parprint('\033[95mpint:\033[0m Version information: '+str(__version__))
                parprint('  uses GPAW '+gpaw.__version__+', ASE '+ase.__version__+' and PHONOPY '+phonopy.__version__)
                parprint('-----------------------------------------------------------------------------')
                parprint('No internet connection available.')
            quit()

        if args.inputfile is not None:
            configpath = os.path.join(os.getcwd(),args.inputfile)
            sys.path.append(os.getcwd())
        else:
            parprint("\033[91mERROR:\033[0m Please use an input file with -i argument.")
            quit()


        if args.geometryfile :
            inFile = os.path.join(os.getcwd(),args.geometryfile)

        if args.drawfigs == True:
            drawfigs = True

        if args.energymeas == True:
            try:
                import pyRAPL
                energymeas = True
                # Start energy consumption calculation.
                pyRAPL.setup()
                meter = pyRAPL.Measurement('dftsolve')
                meter.begin()
            except:
                parprint("\033[91mERROR:\033[0m Unexpected error while using -e argument.")
                parprint("-e works only with Intel CPUs after Sandy Bridge generation. Do not use with AMD CPUs")
                parprint("You also need to install pymongo and pandas libraries.")
                parprint("If you got permission error, try: sudo chmod -R a+r /sys/class/powercap/intel-rapl")
                parprint("More information about the error:")
                parprint(sys.exc_info()[0])
                quit()

    except getopt.error as err:
        # output error, and return with an error code
        parprint (str(err))

    # Start time
    time0 = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(time.time()))

    # Load struct
    struct = struct_from_file(inputfile = configpath, geometryfile = inFile)

    # Write timings of calculation
    with paropen(struct+'-7-Result-Log-Timings.txt', 'a') as f1:
        print("dftsolve.py execution timings (seconds):", end="\n", file=f1)
        print("Execution started:", time0, end="\n", file=f1)

    # Load dftsolve() class
    dftsolver = dftsolve(struct)

    # Run structure calculation
    dftsolver.structurecalc()

    if Optical_calc == False:
        # Run ground state calculation
        dftsolver.groundcalc()

        if Elastic_calc == True:
            # Run elastic calculation
            dftsolver.elasticcalc(drawfigs = drawfigs)

        if DOS_calc == True:
            # Run DOS calculation
            dftsolver.doscalc(drawfigs = drawfigs)

        if Band_calc == True:
            # Run band calculation
            dftsolver.bandcalc(drawfigs = drawfigs)

        if Density_calc == True:
            # Run all-electron density calculation
            dftsolver.densitycalc()    

        if Phonon_calc == True:
            # Run phonon calculation
            dftsolver.phononcalc()  
    else:
        # Run optical calculation
        dftsolver.opticalcalc()

    # Ending of timings
    with paropen(struct+'-7-Result-Log-Timings.txt', 'a') as f1:
        print("---------------------------------------", end="\n", file=f1)

    if args.energymeas == True:
        # Ending of energy consumption measuring.
        meter.end()
        energyresult=meter.result
        with paropen(struct+'-8-Result-Log-Energyconsumption.txt', 'a') as f1:
            print("Energy measurement:-----------------------------------------", end="\n", file=f1)
            print(1e-6*energyresult.duration," Computation time in seconds", end="\n", file=f1)
            print(1e-6*sum(energyresult.pkg)," CPU energy consumption in Joules", end="\n", file=f1)
            print(1e-6*sum(energyresult.dram)," DRAM energy consumption in Joules", end="\n", file=f1)
            print(2.77777778e-7*(1e-6*sum(energyresult.dram)+1e-6*sum(energyresult.pkg))," Total energy consumption in kWh", end="\n", file=f1)
