#!/usr/bin/env python3
"""
mlsolve.py - ML force-field solver

The script accepts a geometry file path and prints a message. Full features
will be added in the future.
"""

import sys
import argparse
from pathlib import Path

try:
    from ase.io import read
except ImportError:
    sys.exit("Error: ASE is required. Install with: pip install ase")

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="MLSolve (early version): load a geometry file."
    )
    parser.add_argument(
        "-g", "--geometry",
        required=True,
        type=str,
        help="Path to input geometry file (cif, xyz, poscar, etc.)"
    )
    return parser.parse_args()

def main():
    args = parse_arguments()
    geom_path = Path(args.geometry)

    if not geom_path.exists():
        sys.exit(f"Error: geometry file '{geom_path}' not found.")

    try:
        atoms = read(geom_path)
    except Exception as e:
        sys.exit(f"Error reading geometry file: {e}")

    print("MLSolve")
    print(f"Loaded structure: {atoms.get_chemical_formula()}")
    print(f"Number of atoms: {len(atoms)}")
    print(f"Cell parameters: {atoms.cell.cellpar().round(3)}")

if __name__ == "__main__":
    main()
