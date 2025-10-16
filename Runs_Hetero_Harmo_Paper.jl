push!(LOAD_PATH, @__DIR__)
using SLH
using Plots
using JLD2
using Statistics
using Random

const σstd = 0.1
const ω0 = 0.0

rng = MersenneTwister(2024)
N = 10000
ω_dist = rand(rng, Normal(ω0, σstd), N)  
const C2 = 2.0
const C1 = -2.0
save_str = "D:/Supplemented_Data/Results/Heterogeneity/C1_$(C1)_σ_$(σstd)_N_$(N)"

u0 = generate_u0(N)
K_list = [1.5,1.3,1.2,1.2,1.0]
for K in K_list
    println("K = $K")
    if !isdir(save_str)
        mkpath(save_str)
    end

    X, t = numerical_simulation_harmo(K, C1, C2, ω_dist, u0; Ttr = 50, Δt = 0.1, T = 200)
    amps, phases = amp_phases_EP(X)

    rho, m_loc, m_dens, separation_ratio = get_rho_distribution(amps, 100, true)
    KOP_A, KOP_P = KOP_s(amps, phases)
    freqs = mean_frequencies(X, t)

    filename = joinpath(save_str, "K_$(K).jld2")
    @save filename amps phases rho KOP_A KOP_P freqs u0
    hist_path = joinpath(save_str, "hist_K_$(K).png")
    plt = histogram(amps, bins=200, xlabel="Amplitude", ylabel="Count", title="Ω = $(mean(freqs)), R = $(KOP_A)")
    savefig(plt, hist_path)
end