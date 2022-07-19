# wham
This repository contains various analysis scripts written for use on the WHAM project.

Some of the more important ones are-

1. eqdsk_analysis_toolbox - Analyze an eqdsk file with contains magnetic field data.
2. genray_analysis_toolbox - Analyze a .nc file created from the ray tracing code Genray.
3. cq3d_analysis_functions - Functions to analyze a .nc file created from the Fokker-Plank code CQL3D.
4. cql3d_analysis_toolbox - Various routines to use the functions defined in cql3d_analysis_functions to parse through a CQL3D output file.
5. charge_sensitive_detector_signal_analysis - Processing and analysis for data from a charge sensitive detector. I wrote this mainly for the WHAM proton detector system but it can be used more generally as well.
6. particle_tracker - Track particles from fusion reactions (MeV scale) through an arbitrary magnetic field geometry defined by an eqdsk. This script can-
    a. Track particles in 3D
    b. Check if particle tracks hit a detector
    c. Implement various detector collimation schemes
