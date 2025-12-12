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


# ---------------------------------------------------------
# Calculators
# ---------------------------------------------------------

def get_ml_calculator(model_type, device="cpu", **kwargs):
    """
    Initializes an ML calculator (MACE only)
    """
    model_type = model_type.lower()

    print(f"\n[Calculators] Requested model: {model_type}, device: {device}")

    if model_type == "mace":
        try:
            from mace.calculators import mace_mp
        except ImportError:
            sys.exit("Error: MACE is not installed. Install with: pip install mace-torch")

        print("  Loading basic MACE model (default).")
        # minimal working configuration
        return mace_mp(model="medium", device=device)

    elif model_type in ("chgnet", "sevennet"):
        print("  Model recognized but not yet implemented.")
        return None

    else:
        print("  Unknown model type. Returning None.")
        return None


# ---------------------------------------------------------
# Task Functions
# ---------------------------------------------------------

def run_static(atoms, config):
    print("\n--- Running static calculation ---")
    calc = get_ml_calculator(config["model"], config["device"], **config)
    atoms.calc = calc

    if calc is None:
        print("No ML calculator available. Static calculation not performed.")
        return

    # Perform a real calculation now
    try:
        pe = atoms.get_potential_energy()
        forces = atoms.get_forces()
        print(f"Potential energy: {pe:.6f} eV")
        print(f"Max force: {forces.max():.6f} eV/Å")
    except Exception as e:
        print(f"Calculation error: {e}")


def run_optimize(atoms, config):
    print("\n--- Running geometry optimization ---")
    print("Optimization not yet implemented.")
    calc = get_ml_calculator(config["model"], config["device"], **config)
    atoms.calc = calc


# ---------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="MLSolve with basic MACE support."
    )
    parser.add_argument(
        "-g", "--geometry",
        required=True,
        type=str,
        help="Path to input geometry file"
    )
    parser.add_argument(
        "-i", "--input",
        required=False,
        type=str,
        help="Configuration dictionary as a string"
    )
    return parser.parse_args()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

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

    # Parse config
    user_config = {}
    if args.input:
        try:
            user_config = ast.literal_eval(args.input)
            if not isinstance(user_config, dict):
                raise ValueError
        except Exception:
            sys.exit("Error: -i must be a valid Python dictionary string.")

    # Evolving default config
    config = {
        "model": "mace",
        "task": "static",
        "device": "cpu",
    }
    config.update(user_config)

    print("MLSolve")
    print(f"Loaded structure : {atoms.get_chemical_formula()}")
    print(f"Number of atoms  : {len(atoms)}")
    print(f"Cell parameters  : {atoms.cell.cellpar().round(3)}")

    print("\nFinal configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # Task dispatcher
    task = config.get("task", "static").lower()

    if task == "static":
        run_static(atoms, config)
    elif task == "optimize":
        run_optimize(atoms, config)
    else:
        print(f"\nError: unknown task '{task}'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
