# Heterogeneous_CGLE
This repository is part of the manuscript "Complete Synchronization and Pattern Selection through Amplitude Dynamics and Diffusion in Heterogeneous Oscillatory Media."

We hope the scripts and data included here will facilitate others in reproducing our results. The repository includes code for simulating both heterogeneous Stuart-Landau oscillators as well as a heterogeneous CGLE under linear global diffusive coupling.


All the scripts were run using a julia v1.11. The package versions can be found in the toml files in the SLH folder. We ran the scripts on Windows 11.


We used the SL_GC_linear_diffusion_2D_Neumann_GC.jl script to numerically integrate a globally coupled CGLE. This was used to produce the data necessary for Figures 4, 5 and 6.
Using the functions in the script, one can perform a parameter ramp, where the amplitude, phase and mean frequency distribution is tracked together with the trivially defined cluster ratio. Parameter values are easily tunable.

We used the Runs_Hetero_Harmo_Paper.jl script to numerically integrate a globally coupled system of heterogeneous SL oscillators. This file uses the defined SLH module. 



