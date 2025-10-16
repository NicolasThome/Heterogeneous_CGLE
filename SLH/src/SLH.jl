module SLH

using Distributions, Roots, HCubature, NLsolve
using KernelDensity, FiniteDiff
using DynamicalSystems, Statistics, LoopVectorization, ChaosTools, DSP
using LsqFit, LinearAlgebra

include("Self_consistency_eq_R_Omega.jl")
include("Guess_amplitude_distributions.jl")
include("SLGCH_N_numerics.jl")
include("Data_analysis.jl")
include("Analytical_Bifurcations.jl")

export self_consistency, find_phase_locked_state, Normal, find_zeros
export predict_radii_distribution, kde, get_rho_distribution, local_maxima
export slgch!, generate_u0, amp_phases_EP, numerical_simulation, KOP_s, mean_frequencies, estimate_period, mean, std, unwrap, instantaneous_frequencies
export sl_harmo_hetero!, numerical_simulation_harmo
export Ω_model_fixed, softplus, nearest_index, nearest, curve_fit, coef  
export BF, SN_solitary, HB_solitary, solve_curve
end