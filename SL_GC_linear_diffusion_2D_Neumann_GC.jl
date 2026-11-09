using DifferentialEquations, Plots, StatsBase, JLD2, Distributions, Random, DSP, Statistics
using KernelDensity: kde

const K = 1.15
d = 0.06
sigma = 0.2
ω_dist = Normal(0.0, sigma)
const c2 = 2.0
const x_min = 0.0
const x_max = 10
const Nx = 100                      
const Lx = x_max - x_min
const dx = Lx / (Nx - 1)

const y_min = 0.0
const y_max = 10
const Ny = 100                    
const Ly = y_max - y_min
const dy = Ly / (Ny - 1)

frequency_seed = 2024
initial_condition_seed = 2025
rng = MersenneTwister(frequency_seed)
const M_ω = rand(rng, ω_dist, Nx, Ny)  

x1 = collect(range(x_min, x_max, length=Nx))
x2 = collect(range(y_min, y_max, length=Ny))

function laplacian_neumann!(lap, W::Array{ComplexF64,2}, dx::Float64, dy::Float64)
    Nx, Ny = size(W)
    for j in 1:Ny
        for i in 1:Nx
            # Second derivative in the x₁-direction:
            if i == 1
                d2Wdx2 = (2*W[2,j] - 2*W[1,j]) / dx^2
            elseif i == Nx
                d2Wdx2 = (2*W[Nx-1,j] - 2*W[Nx,j]) / dx^2
            else
                d2Wdx2 = (W[i-1,j] - 2*W[i,j] + W[i+1,j]) / dx^2
            end
            # Second derivative in the x₂-direction:
            if j == 1
                d2Wdx2_y = (2*W[i,2] - 2*W[i,1]) / dy^2
            elseif j == Ny
                d2Wdx2_y = (2*W[i,Ny-1] - 2*W[i,Ny]) / dy^2
            else
                d2Wdx2_y = (W[i,j-1] - 2*W[i,j] + W[i,j+1]) / dy^2
            end
            lap[i,j] = d2Wdx2 + d2Wdx2_y
        end
    end
    return lap
end

function spatial_mean_2d(W::Array{ComplexF64,2}, dx::Float64, dy::Float64, Lx::Float64, Ly::Float64)
    return (dx * dy * sum(W)) / (Lx * Ly)
end

function pde_rhs_2d_neumann!(dWdt, W, p, t)
    d_val, c1, lap = p  # diffusion, coupling phase, and reusable workspace
    laplacian_neumann!(lap, W, dx, dy)
    W_mean = spatial_mean_2d(W, dx, dy, Lx, Ly)
    Nx, Ny = size(W)
    for j in 1:Ny
        for i in 1:Nx
            term_lin      = (1 - im*M_ω[i,j]) * W[i,j]                     # This is the main correction 
            term_nl       = -(1 + im*c2) * abs2(W[i,j]) * W[i,j]
            term_diff     = d_val * lap[i,j]
            term_coupling = (K + im*K*c1) * (W_mean - W[i,j])
            dWdt[i,j] = term_lin + term_nl + term_diff + term_coupling
        end
    end
end

# KDE locates the two tallest modes; rho counts samples in fixed amplitude windows.
# Manuscript peak rule: 150 KDE grid neighbours, density > 0.05.
# Near grid edges, compare all available neighbours instead of excluding the peak.
# One mode gives 0; invalid data, unresolved modes, or empty windows give NaN.
function rho_from_amplitudes(amp; halfwidth=0.1, peak_neighborhood=150)
    isfinite(halfwidth) && halfwidth > 0 || throw(ArgumentError("halfwidth must be positive and finite"))
    peak_neighborhood >= 1 || throw(ArgumentError("peak_neighborhood must be positive"))
    samples = vec(Float64.(amp))
    isempty(samples) && return NaN
    all(isfinite, samples) || return NaN
    minimum(samples) == maximum(samples) && return 0.0

    estimate = kde(samples)
    xs, ys = estimate.x, estimate.density
    w = peak_neighborhood
    peaks = [i for i in 2:(length(ys)-1)
             if ys[i] > 0.05 && ys[i] > maximum(@view ys[max(1,i-w):i-1]) &&
                ys[i] > maximum(@view ys[i+1:min(length(ys),i+w)])]
    isempty(peaks) && return NaN
    length(peaks) == 1 && return 0.0

    tallest = peaks[sortperm(ys[peaks]; rev=true)[1:2]]
    low, high = sort(xs[tallest])
    # Count both windows independently, including samples in any overlap.
    n_low = count(a -> abs(a - low) < halfwidth, samples)
    n_high = count(a -> abs(a - high) < halfwidth, samples)
    (n_low == 0 || n_high == 0) && return NaN
    return n_high / (n_low + n_high)
end

step   = 0.05
up     = -1.75:step:-1.0         
down   = -1.75:-step:-3.0         
sweeps = Dict(:up=>up, :down=>down)


long_tspan  = (0.0,2000.0)   # very first point of each sweep
short_tspan = (0.0,1000.0)   # all subsequent points
rng_ic = MersenneTwister(initial_condition_seed)
W0_fixed = [rand(rng_ic) + im*rand(rng_ic) for i in 1:Nx, j in 1:Ny]
# One workspace is sufficient for these sequential continuation runs.
lap_buffer = similar(W0_fixed)


results_dict = Dict{Tuple{Float64,Symbol},Any}()
rho_arrays   = Dict{Tuple{Float64,Symbol},Float64}()
mean_frequencies = Dict{Tuple{Float64,Symbol},Any}()

dt_save = 0.1
observation_window = 100.0  # Save only the last 100 time units.
solver_abstol = 1e-7
solver_reltol = 1e-7
rho_halfwidth = 0.1
rho_peak_neighborhood = 150
rho_definition = "KDE two tallest peaks; comparison neighbourhood clipped at grid edges; fixed +/-0.1 windows; N_high/(N_low+N_high); overlap counted in both windows; one mode=0; unresolved=NaN"

for (dir, c1_list) in sweeps
    W_prev = copy(W0_fixed)

    for (i, c1_val) in enumerate(c1_list)
        tspan_use = (i == 1) ? long_tspan : short_tspan
        params = (d, c1_val, lap_buffer)
        println((d, c1_val))


        save_start = max(tspan_use[1], tspan_use[2] - observation_window)
        saveat_times = save_start:dt_save:tspan_use[2]

        prob = ODEProblem(pde_rhs_2d_neumann!, W_prev, tspan_use, params)
        sol = solve(prob, Tsit5(); dt=10.0, abstol=solver_abstol, reltol=solver_reltol,
                    saveat=saveat_times, save_start=false, save_end=true, dense=false,
                    maxiters=10_000_000)

        t_points = sol.t
        n_t = length(t_points)
        mean_frequencies = Array{Float64}(undef, Nx, Ny)

        for i in 1:Nx
            for j in 1:Ny

                oscillator_vals = [sol.u[k][i,j] for k in 1:n_t]

                phases = angle.(oscillator_vals)
                phases_unwrapped = unwrap(phases)

                A = [t_points ones(n_t)]
                coeff = A \ phases_unwrapped
                mean_frequencies[i,j] = coeff[1]  
            end
        end


        W_final = sol.u[end]
        amp = abs.(W_final)
        rho = rho_from_amplitudes(amp; halfwidth=rho_halfwidth,
                                  peak_neighborhood=rho_peak_neighborhood)
        if isnan(rho)
            @warn "KDE rho is undefined; inspect the amplitude distribution" C1=c1_val branch=dir
        end

        key = (c1_val, dir)
        rho_arrays[key] = rho
        results_dict[key] = (
            W_final = W_final,
            amp = amp,
            t_final = tspan_use[end],
            mean_frequency = mean_frequencies  # Now a matrix of frequencies
        )
        W_prev = W_final
    end
end

rho_data = vcat([rho_arrays[(c1, :up)] for c1 in up],
                [rho_arrays[(c1, :down)] for c1 in down])
diff_C1_vals_full = vcat(collect(up), collect(down))

Save_loc = joinpath(@__DIR__, "Supplemented_Data", "Results", "GC_SL_2D_Neumann")
mkpath(Save_loc)
output_file = joinpath(Save_loc, "results_adiabatic_d0p06_sigma0p2.jld2")
@save output_file results_dict rho_arrays rho_data rho_definition rho_halfwidth rho_peak_neighborhood diff_C1_vals_full up down x1 x2 short_tspan long_tspan K d sigma c2 M_ω W0_fixed frequency_seed initial_condition_seed dt_save observation_window solver_abstol solver_reltol




