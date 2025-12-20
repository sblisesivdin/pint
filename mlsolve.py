#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mlsolve.py: MLFF Solver with MACE, CHGNet and SevenNet Support.

This script performs calculations using machine-learning potentials by only requesting
a geometry file (-g) and a parameter dictionary (-i) from the user.

Supported Models:
  1. MACE (Multi-Atomic Cluster Expansion)
  2. CHGNet (Charge-Informed Graph Neural Network)
  3. SevenNet (Scalable Equivariance Enabled Neural Network)

Requirements:
  - ase, torch, numpy
  - mace-torch (for MACE)
  - chgnet (for CHGNet)
  - sevenn (for SevenNet)

Usage Example:
  python mlsolve.py -g structure.cif -i "{'model': 'mace', 'task': 'optimize', 'fmax': 0.01"

License: MIT
"""

import sys
import argparse
import ast
import time
import warnings
import numpy as np
from pathlib import Path

# ASE Libraries
try:
    from ase import Atoms
    from ase.io import read, write
    from ase.optimize import BFGS, FIRE, LBFGS
    from ase.constraints import ExpCellFilter, UnitCellFilter
except ImportError:
    sys.exit("Error: ASE (Atomic Simulation Environment) library not found.")

# Suppress warnings (clean output)
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# SECTION 1: CALCULATOR FACTORY
# -----------------------------------------------------------------------------

def get_ml_calculator(model_type, device='cpu', **kwargs):
    """
    Initializes and returns an ASE calculator for the requested ML model.
    Required libraries are imported lazily.

    Arguments:
        model_type (str): 'mace', 'chgnet' or 'sevennet'.
        device (str): 'cpu', 'cuda', or 'mps'.
        **kwargs: Additional model-specific parameters (e.g., model_path, dtype).
    
    Returns:
        ase.calculator.Calculator: Initialized calculator instance.
    """
    model_type = model_type.lower()
    
    # --- MACE Configuration ---
    if model_type == 'mace':
        try:
            from mace.calculators import mace_mp, mace_off
            # Variant selection (small, medium, large). Default: medium
            variant = kwargs.get('variant', 'medium')
            # Data type (float32 uses less memory and is usually enough)
            dtype = kwargs.get('dtype', 'float64')
            
            print(f"--> Loading MACE Model ({variant}) - Device: {device}, Type: {dtype}...")
            
            # mace_off: for organic molecules, mace_mp: for inorganic crystals
            if kwargs.get('organic', False):
                return mace_off(model=variant, device=device, default_dtype=dtype)
            else:
                # dispersion=False by default; user may enable it
                dispersion = kwargs.get('dispersion', False)
                return mace_mp(model=variant, device=device, default_dtype=dtype, dispersion=dispersion)
        except ImportError:
            sys.exit("Error: MACE library is not installed. Install with: pip install mace-torch")

    # --- CHGNet Configuration ---
    elif model_type == 'chgnet':
        try:
            from chgnet.model import CHGNet
            from chgnet.model.model import CHGNetCalculator
            
            print(f"--> Loading CHGNet Model - Device: {device}...")
            
            model_path = kwargs.get('model_path', None)
            
            if model_path:
                model = CHGNet.from_file(model_path)
            else:
                model = CHGNet.load()
            
            return CHGNetCalculator(model=model, use_device=device)
        except ImportError:
            sys.exit("Error: CHGNet library is not installed. Install with: pip install chgnet")

    # --- SevenNet Configuration ---
    elif model_type == 'sevennet':
        try:
            from sevenn.sevennet_calculator import SevenNetCalculator
            
            print(f"--> Loading SevenNet Model - Device: {device}...")
            # Default model: 7net-0 (Materials Project based)
            model_name = kwargs.get('model_name', '7net-0')
            
            return SevenNetCalculator(model=model_name, device=device)
        except ImportError:
            sys.exit("Error: SevenNet library is not installed. Install with: pip install sevenn")

    else:
        sys.exit(f"Error: Unknown model type '{model_type}'. Supported options: mace, chgnet, sevennet")

# -----------------------------------------------------------------------------
# SECTION 2: ARGUMENT PARSING & MAIN LOGIC
# -----------------------------------------------------------------------------

def parse_arguments():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="MLSolve: Unified Interface for MACE, CHGNet, and SevenNet",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('-g', '--geometry', type=str, required=True, 
                        help='Path to input geometry file (cif, xyz, poscar, traj etc.)')
    parser.add_argument('-i', '--input', type=str, required=True, 
                        help="Configuration dictionary (as a string).\n"
                             "Example: \"{'model': 'mace', 'fmax': 0.05, 'task': 'optimize'}\"")
    return parser.parse_args()

def main():
    # Start time
    t0 = time.time()
    
    # 1. Load arguments
    args = parse_arguments()
    
    # 2. Safely parse dictionary string using ast.literal_eval
    try:
        user_params = ast.literal_eval(args.input)
        if not isinstance(user_params, dict):
            raise ValueError
    except (ValueError, SyntaxError):
        sys.exit("Error: -i argument must be a valid Python dictionary.\n"
                 "Example usage: -i \"{'model': 'chgnet', 'device': 'cpu'}\"")

    # 3. Default configuration and override with user input
    config = {
        'model': 'mace',           # Default model
        'task': 'optimize',        # Task: 'optimize', 'static', 'md' (future)
        'device': 'cpu',          # Hardware acceleration: 'cuda', 'cpu', 'mps'
        'fmax': 0.05,              # Force convergence criterion (eV/Å)
        'steps': 200,              # Max optimization steps
        'cell_relax': True,        # Relax the unit cell? (Variable Cell Relaxation)
        'optimizer': 'BFGS',       # Algorithm: 'BFGS' or 'FIRE'
        'trajectory': 'out.traj',  # Output trajectory file
        'logfile': 'mlsolve.log',  # Log file
        'out_file': 'optimized.cif' # Output structure file
    }
    config.update(user_params)

    print("=========================================================")
    print(f"  MLSolve - {time.ctime()}")
    print("=========================================================")
    print(f"  Geometry File   : {args.geometry}")
    print(f"  Model           : {config['model'].upper()}")
    print(f"  Task            : {config['task']}")
    print(f"  Device          : {config['device']}")
    print("=========================================================\n")

    # 4. Read geometry
    input_path = Path(args.geometry)
    if not input_path.exists():
        sys.exit(f"Error: file '{input_path}' not found.")
    
    try:
        atoms = read(input_path)
        print(f"Structure Loaded: {atoms.get_chemical_formula()}")
        print(f"Number of Atoms: {len(atoms)}")
        print(f"Cell           : {atoms.cell.cellpar().round(3)}\n")
    except Exception as e:
        sys.exit(f"Error: Geometry file could not be read. ASE error: {e}")

    # 5. Initialize and attach the calculator
    calc = get_ml_calculator(
        config['model'], 
        device=config['device'], 
        **config
    )
    atoms.calc = calc

    # 6. Execute task
    if config['task'] == 'static':
        print("--- Starting Static Calculation ---")
        try:
            pe = atoms.get_potential_energy()
            forces = atoms.get_forces()
            fmax = np.max(np.linalg.norm(forces, axis=1))
            
            print(f"Potential Energy : {pe:.6f} eV")
            print(f"Max Force        : {fmax:.6f} eV/Å")
            
            # CHGNet-specific output (magnetic moments)
            if config['model'] == 'chgnet':
                try:
                    mag = atoms.get_magnetic_moments()
                    print(f"Magnetic Moments (first 5 atoms): {mag[:5]}...")
                except Exception as e:
                    print(f"Warning: Could not retrieve magnetic moments: {e}")
        except Exception as e:
            print(f"Calculation Error: {e}")

    elif config['task'] == 'optimize':
        print(f"--- Starting Geometry Optimization ({config['optimizer']}) ---")
        print(f"Target fmax: {config['fmax']} eV/Å")
        
        # Relaxation target: atomic positions + cell or only positions
        if config['cell_relax']:
            print("Info: Relaxing both atomic positions and unit cell (ExpCellFilter).")
            ecf = ExpCellFilter(atoms)
            opt_target = ecf
        else:
            print("Info: Relaxing atomic positions only (Fixed cell).")
            opt_target = atoms

        # Optimizer selection
        if config['optimizer'].upper() == 'FIRE':
            dyn = FIRE(opt_target, trajectory=config['trajectory'], logfile=config['logfile'])
        else:
            dyn = BFGS(opt_target, trajectory=config['trajectory'], logfile=config['logfile'])

        try:
            dyn.run(fmax=config['fmax'], steps=config['steps'])
            
            write(config['out_file'], atoms)
            print(f"\nOptimization completed.")
            print(f"Final Energy : {atoms.get_potential_energy():.6f} eV")
            print(f"Final Cell   : {atoms.cell.cellpar().round(3)}")
            print(f"Output       : {config['out_file']}")
            print(f"Trajectory   : {config['trajectory']}")
            
        except Exception as e:
            print(f"\nError during optimization: {e}")
            write('crash_dump.cif', atoms)
            print("Current structure saved to 'crash_dump.cif'.")

    # 7. End
    elapsed = time.time() - t0
    print(f"\nTotal Time: {elapsed:.2f} seconds")
    print("Done.")

if __name__ == "__main__":
    main()
