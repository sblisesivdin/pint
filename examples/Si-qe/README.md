# Silicon Quantum ESPRESSO Conversion Example

This directory provides a minimal Quantum ESPRESSO `pw.x` input for silicon to convert configuration to dftolve.py's input file by using `converters/qeconverter.py`.

To regenerate the `dftsolve.py` inputs run:

    qeconverter.py --input si.scf.in --output-dir Si-qe --system-name Silicon

Then execute `dftsolve.py` using the produced files:

    mpirun -np 4 dftsolve.py -i Silicon.py -g Silicon.cif

