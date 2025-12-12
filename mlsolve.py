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
        description="MLSolve: geometry + task scaffolding."
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
        help="Configuration dictionary as a string, e.g. \"{'task': 'static'}\""
    )
    return parser.parse_args()


def run_static(atoms, config):
    """Placeholder for the future static calculation."""
    print("\n--- Running static calculation (placeholder) ---")
    print("No ML calculator attached yet.")
    print("This will compute energy & forces.")


def run_optimize(atoms, config):
    """Placeholder for the future geometry optimization."""
    print("\n--- Running geometry optimization (placeholder) ---")
    print("No optimizer or ML calculator yet.")
    print("This will perform relaxation.")


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

    # Parse user configuration dictionary
    user_config = {}
    if args.input:
        try:
            user_config = ast.literal_eval(args.input)
            if not isinstance(user_config, dict):
                raise ValueError
        except Exception:
            sys.exit("Error: -i must be a valid Python dictionary string.")

    # Default configuration grows slowly as we evolve
    config = {
        "model": "mace",
        "task": "static",
        "device": "cpu"
    }

    # Merge user config
    config.update(user_config)

    print("MLSolve")
    print(f"Loaded structure : {atoms.get_chemical_formula()}")
    print(f"Number of atoms  : {len(atoms)}")
    print(f"Cell parameters  : {atoms.cell.cellpar().round(3)}")

    print("\nFinal configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # --- Task dispatching ---
    task = config.get("task", "static").lower()

    if task == "static":
        run_static(atoms, config)
    elif task == "optimize":
        run_optimize(atoms, config)
    else:
        print(f"\nError: unknown task '{task}'.")
        print("Supported tasks: static, optimize")
        sys.exit(1)


if __name__ == "__main__":
    main()
