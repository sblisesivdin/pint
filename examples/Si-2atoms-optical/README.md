# Example: 2 Atoms Silicon Calculations (3 Steps)

This example has three steps. First step is the ground state, DOS and band structure calculations. And second and third steps are the calculations of optical properties. Please do not forget to run optical calculations seperately.

To run the first step of calculation with MPI please execute the following command in this folder.

    mpirun -np 4 dftsolve.py -i Si-Step1-ground_dos_band.py -g Si_mp-149_primitive_Example.cif
    
or

    gpaw -P4 python ~/path-to-pint/dftsolve.py -- -i Si-Step1-ground_dos_band.py -g Si_mp-149_primitive_Example.cif

Here, ~/path-to-pint shows a full path your Pint folder.
	
And then for the second step, there are two possibilities. Real and imaginary parts of dielectric function are usually calculated with random phase approximation (RPA). With GPAW, we can go beyond the RPA using the Bethe-Salpeter equation (BSE).

In this example, the second step is the optical properties calculation with RPA method. Optical calculations uses too much RAM. Here, our input is very easy and it is not important to run it with one or more cores. We are running the code on a single-core as:

    dftsolve.py -i Si-Step2-optical-RPA.py -g Si_mp-149_primitive_Example.cif

and, the third step is the optical properties calculation with BSE method. It can be executed as:

    dftsolve.py -i Si-Step3-optical-BSE.py -g Si_mp-149_primitive_Example.cif

To use more cores to calculate, firstly observe your calculation's RAM usage with command `htop` or with a similar command. Then you can use (for example 4 cores) :

   mpirun -np 4 dftsolve.py -i Si-Step2-optical-RPA.py -g Si_mp-149_primitive_Example.cif
   mpirun -np 4 dftsolve.py -i Si-Step3-optical-BSE.py -g Si_mp-149_primitive_Example.cif
   
or

    gpaw -P4 python ~/path-to-pint/dftsolve.py -- -i Si-Step2-optical-RPA.py -g Si_mp-149_primitive_Example.cif
    gpaw -P4 python ~/path-to-pint/dftsolve.py -- -i Si-Step3-optical-BSE.py -g Si_mp-149_primitive_Example.cif

But for this example, it will be completed in a few seconds even with a single core. 

Output files of optical calculations will be named accordingly.
