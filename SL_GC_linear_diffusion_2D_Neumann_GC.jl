using DifferentialEquations, Plots, StatsBase, JLD2, LaTeXStrings, Distributions, Random, DSP, Statistics

K  = 1.2
d = 0.03
ω_dist = Normal(0.0, 0.3)
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

rng = MersenneTwister(2024)
M_ω = rand(rng, ω_dist, Nx, Ny)  

x1 = collect(range(x_min, x_max, length=Nx))
x2 = collect(range(y_min, y_max, length=Ny))

function laplacian_neumann(W::Array{ComplexF64,2}, dx::Float64, dy::Float64)
    Nx, Ny = size(W)
    lap = similar(W)
    for i in 1:Nx
        for j in 1:Ny
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
    d_val, c1 = p  # the diffusion coefficient
    lap = laplacian_neumann(W, dx, dy)
    W_mean = spatial_mean_2d(W, dx, dy, Lx, Ly)
    Nx, Ny = size(W)
    for i in 1:Nx
        for j in 1:Ny
            term_lin      = W[i,j] - im*M_ω[i,j]
            term_nl       = -(1 + im*c2) * abs2(W[i,j]) * W[i,j]
            term_diff     = d_val * lap[i,j]
            term_coupling = (K + im*K*c1) * (W_mean - W[i,j])
            dWdt[i,j] = term_lin + term_nl + term_diff + term_coupling
        end
    end
end

step   = 0.05
up     = -1.75:step:-1.0         
down   = -1.75:-step:-3.0         
sweeps = Dict(:up=>up, :down=>down)


long_tspan  = (0.0,1000.0)   # very first point of each sweep
short_tspan = (0.0,250.0)   # all subsequent points
W0_fixed = [rand() + im*rand() for i in 1:Nx, j in 1:Ny]


results_dict = Dict{Tuple{Float64,Symbol},Any}()
rho_arrays   = Dict{Tuple{Float64,Symbol},Float64}()
mean_frequencies = Dict{Tuple{Float64,Symbol},Any}()

dt_save = 0.5  # Time step for saving during last 100 time units

for (dir, c1_list) in sweeps
    W_prev = copy(W0_fixed)

    for (i, c1_val) in enumerate(c1_list)
        tspan_use = (i == 1) ? long_tspan : short_tspan
        params = (d, c1_val)
        println(params)


        save_start = max(tspan_use[1], tspan_use[2] - 50.0)
        saveat_times = save_start:dt_save:tspan_use[2]

        prob = ODEProblem(pde_rhs_2d_neumann!, W_prev, tspan_use, params)
        sol = solve(prob, Tsit5(); dt=10.0, abstol=1e-6, reltol=1e-6, saveat=saveat_times)

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
        mid = mean(amp)
        sync = minimum(amp) - maximum(amp)
        rho = (abs(sync) < 1e-3) ? 1.0 : sum(amp .< mid) / (Nx * Ny)

        key = (c1_val, dir)
        rho_arrays[key] = rho
        results_dict[key] = (
            W_final = W_final,
            amp = amp,
            t_final = tspan_use[end],
            mean_frequency = mean_frequencies  # Now a matrix of frequencies
        )
        #W_prev = W_final
    end
end

rho_data = [rho_arrays[(c1, :up)]   for c1 in up  ] ∪
           [rho_arrays[(c1, :down)] for c1 in down]
diff_C1_vals_full = vcat(collect(up), collect(down))

Save_loc = "Supplemented_Data/Results/GC_SL_2D_Neumann/"
@save Save_loc * "Supplemented_Data/Results/GC_SL_2D_Neumann/" results_dict rho_arrays diff_C1_vals_full up down x1 x2 short_tspan long_tspan




