Heterogeneous_CGLE

This repository accompanies the manuscript
“Complete Synchronization and Pattern Selection through Amplitude Dynamics and Diffusion in Heterogeneous Oscillatory Media.”

It provides all scripts required to reproduce the numerical results presented in the paper. The repository contains implementations for simulating both heterogeneous Stuart–Landau oscillators and a heterogeneous complex Ginzburg–Landau equation (CGLE) with linear global diffusive coupling.
The data used in the generation of the plots has been uploaded to Zenodo with the DOI xyz.

Environment:

All simulations were performed using Julia v1.11 on Windows 11.
The exact package versions used are specified in the Project.toml and Manifest.toml files located in the SLH folder.


Contents: 
1. Globally Coupled CGLE

The script SL_GC_linear_diffusion_2D_Neumann_GC.jl numerically integrates the globally coupled CGLE.
It was used to generate the data shown in Figures 4, 5, and 6 of the manuscript.

Using the functions provided in this script, one can perform parameter ramps in which the amplitude, phase, and mean frequency distributions are tracked simultaneously with the trivially defined cluster ratio.
All relevant parameters can be easily adjusted within the script.

2. Globally Coupled Stuart–Landau Oscillators

The script Runs_Hetero_Harmo_Paper.jl numerically integrates a globally coupled system of heterogeneous Stuart–Landau oscillators. It was used to generate the data shown in Figure 7 of the manuscript.
It relies on the SLH module included in the repository.

3. Center Manifold analysis

To compare the center manifold approach to nuermical data, we used the Figure_8.jl file


