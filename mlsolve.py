#!/usr/bin/env python3
"""
mlsolve.py - ML force-field solver

The script accepts a geometry file path and parses the args, and prints a message. Full features
will be added in the future.
"""
import sys
import argparse
import ast
from pathlib import Path

try:
    from ase.io import read
except ImportError:
    sys.exit("Error: ASE is required. Install with: pip install ase")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="MLSolve: load geometry and parse config."
    )
    parser.add_argument(
        "-g", "--geometry",
        required=True,
        type=str,
        help="Path to input geometry file (cif, xyz, poscar, etc.)"
    )
    parser.add_argument(
        "-i", "--input",
        required=False,
        type=str,
        help="Configuration dictionary as a string, e.g. \"{'model': 'mace'}\""
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    geom_path = Path(args.geometry)

    # Geometry loading
    if not geom_path.exists():
        sys.exit(f"Error: geometry file '{geom_path}' not found.")

    try:
        atoms = read(geom_path)
    except Exception as e:
        sys.exit(f"Error reading geometry file: {e}")

    # Parse configuration dictionary
    user_config = {}
    if args.input:
        try:
            user_config = ast.literal_eval(args.input)
            if not isinstance(user_config, dict):
                raise ValueError
        except Exception:
            sys.exit("Error: -i must be a valid Python dictionary string.")

    print("MLSolve")
    print(f"Loaded structure : {atoms.get_chemical_formula()}")
    print(f"Number of atoms  : {len(atoms)}")
    print(f"Cell parameters  : {atoms.cell.cellpar().round(3)}")

    if user_config:
        print("\nUser configuration:")
        for k, v in user_config.items():
            print(f"  {k}: {v}")
    else:
        print("\nNo user configuration provided.")


if __name__ == "__main__":
    main()
