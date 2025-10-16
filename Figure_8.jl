using Plots, DifferentialEquations, OrdinaryDiffEq, FFMPEG,StatsBase, JLD2
gr()
function pde_rhs!(du, u, p, t)
    λ, A, B, C, D, dx = p
    N = length(u)
    x²mean = mean(u .^ 2)
    x3mean = mean(u .^ 3)

    # Neumann Laplacian
    du[1] = λ*u[1] + A*(u[1]^2-x²mean) + B*(u[1]^3-x3mean) + C*x²mean*u[1] + D*(u[2] - u[1]) / dx^2
    for i in 2:N-1
        du[i] = λ*u[i] + A*(u[i]^2-x²mean) + B*(u[i]^3-x3mean) + C*x²mean*u[i] +
                D*(u[i-1] - 2u[i] + u[i+1]) / dx^2
    end
    du[N] = λ*u[N] + A*(u[N]^2-x²mean) + B*(u[N]^3-x3mean) + C*x²mean*u[N] + D*(u[N-1] - u[N]) / dx^2
end

function sigmoid_initial_condition(x; a = 0.05, x0 = 5, w = 0.5, ϵ = 0.001)
    base = -a * tanh.((x .- x0) ./ w)
    noise = ϵ * randn(length(x))  # Gaussian noise with std dev ϵ
    return base .+ noise
end

sqrtarg(ρ; A, B, C, λ) =
    -4*B*λ + A^2 * (1-2ρ)^2 - 4*(3B-C)*λ*(-1+ρ)*ρ

function u1_fp_a(ρ; A, B, C, λ)
    Δ = sqrtarg(ρ; A, B, C, λ)
    if Δ < 0 || isnan(Δ)
        return NaN
    end
    num   = (-1+ρ)*(A*(-1+2ρ)-sqrt(Δ))
    denom = 2*(B+3B*(-1+ρ)*ρ-C*(-1+ρ)*ρ)
    return -num / denom
end

# b ↔ + sign (not used here, but kept for completeness)
function u1_fp_b(ρ; A, B, C, λ)
    Δ = sqrtarg(ρ; A, B, C, λ)
    println(Δ)
    if Δ < 0 || isnan(Δ)
        return NaN
    end
    num   = (-1+ρ)*(A*(-1+2ρ)+sqrt(Δ))
    denom = 2*(B+3B*(-1+ρ)*ρ-C*(-1+ρ)*ρ)
    return -num/denom
end
# -- Spatial discretization
L = 10                      # domain length
Nx = 800                     # number of spatial points
dx = L / (Nx - 1)            # spatial step
xgrid = range(0, L; length=Nx)

u0 = sigmoid_initial_condition(xgrid; x0=5, w=0.5, ϵ=0)
#plot(xgrid,sigmoid_initial_condition(xgrid; ϵ=0))
u0 .-= mean(u0)  # enforce ⟨u⟩ = 0
#plot(xgrid,u0)


λ = 0.00005     # bifurcation parameter λ₊
A = 0.1
B = -2 / (2*sqrt(3) - 3)
C = -1
D = 0.000001      # diffusion coefficient

np = 50 
                
A_path1 = ones(np).*-0.05
λ_path1 = LinRange(-0.00014, 0.00005, np)

A_path2 = LinRange(-0.05, 0.05, 2*np)
λ_path2 = ones(2*np).*0.00005

A_path3 = ones(np).*0.05
λ_path3 = LinRange(0.00005, -0.00014, np)

A_path = vcat(A_path1,A_path2,A_path3)
λ_path = vcat(λ_path1,λ_path2,λ_path3)
#A_vals = collect(-0.25) 
rho_vals = Float64[]

for (Ai, λi) in zip(A_path, λ_path)
    p = (λi, Ai, B, C, D, dx)
    tspan = (0.0, 800000.0)
    prob = ODEProblem(pde_rhs!, u0, tspan, p)
    sol = solve(prob, Tsit5();  reltol=1e-8, abstol=1e-8)

    xi = sol.u[end]
    meanxi = (maximum(xi)+minimum(xi))/2
    
    high_amp = count(x -> x > meanxi, xi)

    rho1 = high_amp / length(xgrid)
    plt=plot(xgrid, xi, title="A = $Ai", lw=3)
    # Evaluate fixed points
    ρ = rho1
    u1a = u1_fp_a(ρ; A=Ai, B, C, λ=λi)
    u2a = -ρ / (1 - ρ) * u1a

    u1b = u1_fp_b(ρ; λ=λi, A=Ai, B=B, C=C)
    u2b = -ρ / (1 - ρ) * u1b
    # Construct spatial profiles
    ua = [xi < ρ * L ? u1a : u2a for xi in xgrid]
    #ub = [xi < ρ * L ? u1b : u2b for xi in xgrid]

    # To check !
    # plot!(plt,xgrid, ua, lw=2, label="u_1^a", xlabel="x", ylabel = "u(x)")
    # #plot!(plt,xgrid, ub, lw=2, label="u_1^b")
    # display(plt)
    push!(rho_vals, rho1)
end
plt=scatter(1:length(rho_vals), rho_vals)
# Stepwise energy function
function energy_2cluster(ρ; A, B, C, λ)
    if ρ ≤ 0 || ρ ≥ 1
        return NaN
    end

    u1 = u1_fp_a(ρ; A=A, B=B, C=C, λ=λ)
    if isnan(u1)
        return NaN
    end
    u2 = -ρ * u1 / (1 - ρ)

    u1², u1³, u1⁴ = u1^2, u1^3, u1^4
    u2², u2³, u2⁴ = u2^2, u2^3, u2^4

    mean_u2 = ρ * u1² + (1 - ρ) * u2²
    E = - (λ / 2) * (ρ * u1² + (1 - ρ) * u2²) -
        (A / 3) * (ρ * u1³ + (1 - ρ) * u2³) -
        (B / 4) * (ρ * u1⁴ + (1 - ρ) * u2⁴) -
        (C / 2) * mean_u2^2

    return E
end


rhovals = 0.00001:0.001:0.99999
ρstars = Float64[]

for (Ai, λi) in zip(A_path, λ_path)
    Evals = [energy_2cluster(ρ; A=Ai, B=B, C=C, λ=λi) for ρ in rhovals]
    idx_min = argmin(Evals)

    valid_idxs = findall(!isnan, Evals)
    valid_Evals = Evals[valid_idxs]
    valid_ρs = rhovals[valid_idxs]

    min_index = argmin(valid_Evals)
    min_ρ = valid_ρs[min_index]
    push!(ρstars, min_ρ)

end


using Plots

function plot_contour_with_simulation(
    rho_vals, ρstars, A_path, λ_path,
    λ_path1, A_path2, λ_path3,
    A_vals_Solitary, f_vals_Solitary;
    save_as::Union{String, Nothing} = nothing
)
    # --- Composite x-axis setup ---
    n1 = length(λ_path1)
    n2 = length(A_path2)
    n3 = length(λ_path3)
    xvals = 1:(n1 + n2 + n3)

    # --- Top plot: ρ* vs composite parameter ---
    plt = scatter(xvals, rho_vals;
        label = "numeric simulations", lw = 2,
        xlabel = "", ylabel = "ρ*")

    plot!(plt, xvals, ρstars;
        label = "F[ρ,u]", lw = 2,
        ylabel = "ρ*")

    # Segment dividers
    vline!([n1, n1 + n2], linestyle = :dash, color = :gray, label = "")

    # Annotations for λ, A, λ
    loc_annot = minimum(ρstars) - 0.1 * (maximum(ρstars) - minimum(ρstars))
    annotate!(plt, [
        (n1 / 2, loc_annot, text("λ", :black, 10, :center)),
        (n1 + n2 / 2, loc_annot, text("A", :black, 10, :center)),
        (n1 + n2 + n3 / 2, loc_annot, text("λ", :black, 10, :center)),
    ])

    plot!(plt, xticks = :none)
    plot!(plt, bottom_margin = 5Plots.mm)

    # --- Bottom plot: λ vs A ---
    Plt2 = plot(A_vals_Solitary, f_vals_Solitary;
         xlabel = "A",
         ylabel = "λ",
         title = "",
         label = "SN_Solitary",
         lw = 2)

    plot!(Plt2, A_path, λ_path, label = "C")
    plot!(Plt2, A_vals_Solitary, zeros(length(A_vals_Solitary)), label = "BF", lw = 3, color = :black)

    # Arrows along contour
    arrow_indices = [10, length(A_path) ÷ 2, length(A_path) - 10]

    for idx in arrow_indices
        dx = A_path[idx+1] - A_path[idx]
        dy = λ_path[idx+1] - λ_path[idx]
        quiver!(Plt2, [A_path[idx]], [λ_path[idx]],
                quiver = ([dx], [dy]),
                arrow = true, color = :orange, linewidth = 1.5)
    end

    # --- Combine plots ---
    final = plot(plt, Plt2, layout = (2, 1))

    # Optional save
    if save_as !== nothing
        savefig(final, save_as)
    end

    return final
end

plot_contour_with_simulation(
    rho_vals, ρstars, A_path, λ_path,
    λ_path1, A_path2, λ_path3,
    A_vals_Solitary, f_vals_Solitary;
    save_as = "contour.png"
)

function save_contour_plot_data(
    filename::String,
    rho_vals, ρstars,
    A_path, λ_path,
    λ_path1, A_path2, λ_path3,
    A_vals_Solitary, f_vals_Solitary
)
    data = Dict(
        "rho_vals" => rho_vals,
        "ρstars" => ρstars,
        "A_path" => A_path,
        "λ_path" => λ_path,
        "λ_path1" => λ_path1,
        "A_path2" => A_path2,
        "λ_path3" => λ_path3,
        "A_vals_Solitary" => A_vals_Solitary,
        "f_vals_Solitary" => f_vals_Solitary
    )

    @save filename data
    println("✅ Data saved to '$filename'")
end
save_loc="Some_String"
save_contour_plot_data(
    save_loc*"contour_plot_data.jld2",
    rho_vals, ρstars,
    A_path, λ_path,
    λ_path1, A_path2, λ_path3,
    A_vals_Solitary, f_vals_Solitary
)



