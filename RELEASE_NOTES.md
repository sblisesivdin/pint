## Release Notes

### Development Version

- Initializing context and tools for the session.
- Removed gg.py and gui_files/ directory.
- Renaming project and its files as: gpaw-tools -> Pint, gpawsolve.py -> dftsolve.py, and asapsolve -> mdsolve.py
- Replace the gpaw-tools setup with the Pint setup
- Implement an ML solver script: mlsolve.py
- Remove GW calculations and related parameters from dftsolve.py
- Simplify GPW file writing logic (Always mode="all")
- A major refactorization of global variable usage to dataclass! This change ensures type safety, centralized configuration, reduced global namespace pollution, and improved code organization with total backward compatibility.
- In addition to the refactorization of global variable usage, more than 30 security warnings have been fixed.
