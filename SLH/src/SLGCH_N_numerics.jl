

function slgch!(du, u, p, t)
    K, C1, C2i = p
    len2 = length(u)
    N    = len2 >> 1

    sx = zero(eltype(u))
    sy = zero(eltype(u))
    @avx for i ∈ 1:2:len2
        sx += u[i]
        sy += u[i+1]
    end
    invN = one(eltype(u)) / N
    mx   = sx * invN
    my   = sy * invN

    @avx for j ∈ 1:N
        i1 = (j << 1) - 1
        xi = u[i1]
        yi = u[i1+1]

        Dx = mx - xi
        Dy = my - yi
        r2 = xi*xi + yi*yi
        c  = C2i[j]*r2

        du[i1]     = xi - r2*xi + c*yi + K*(Dx - C1*Dy)
        du[i1 + 1] = yi - r2*yi - c*xi + K*(C1*Dx + Dy)
    end
    return nothing
end

function sl_harmo_hetero!(du, u, p, t)
    K, C1, C2, ωi = p
    len2 = length(u)
    N    = len2 >> 1

    sx = zero(eltype(u))
    sy = zero(eltype(u))
    @avx for i ∈ 1:2:len2
        sx += u[i]
        sy += u[i+1]
    end
    invN = one(eltype(u)) / N
    mx   = sx * invN
    my   = sy * invN

    @avx for j ∈ 1:N
        i1 = (j << 1) - 1
        xi = u[i1]
        yi = u[i1+1]

        Dx = mx - xi
        Dy = my - yi
        r2 = xi*xi + yi*yi
        c  = C2*r2
        ω = ωi[j]

        du[i1]     = xi - ω*yi - r2*xi + c*yi + K*(Dx - C1*Dy)
        du[i1 + 1] = yi + ω*xi- r2*yi - c*xi + K*(C1*Dx + Dy)
    end
    return nothing
end

generate_u0(N::Integer) = 2 .* rand(2*N) .- 1.1

function amp_phases_EP(X::StateSpaceSet)                
    reals = X[end][1:2:end]         
    imags = X[end][2:2:end]        
    amps  = sqrt.(reals .^ 2 .+ imags .^ 2)
    phases = mod2pi.(angle.(reals .+ im.*imags))
    return(amps, phases)
end

function numerical_simulation(K::Float64, C1::Float64, c_dist::Distribution{Univariate,Continuous}, N::Int; Ttr::Int = 50, Δt::Float64 = 0.1, T::Int = 20)
    C2i = rand(c_dist, N)
    p0 = (K, C1, C2i)
    u0 = generate_u0(N)
    ds = CoupledODEs(slgch!, u0, p0)
    X, t = trajectory(ds, T; Ttr, Δt)
    return(X, t)
end

function numerical_simulation(K::Float64, C1::Float64, c_dist::Vector{Any}, u0_given ; Ttr::Int = 50, Δt::Float64 = 0.1, T::Int = 20)
    C2i = c_dist
    p0 = (K, C1, C2i)
    u0 = u0_given
    ds = CoupledODEs(slgch!, u0, p0)
    X, t = trajectory(ds, T; Ttr, Δt)
    return(X, t)
end

function numerical_simulation_harmo(K::Float64, C1::Float64,C2::Float64, ω_dist::Distribution{Univariate,Continuous}, N::Int; Ttr::Int = 50, Δt::Float64 = 0.1, T::Int = 20)
    ωi = rand(ω_dist, N)
    p0 = (K, C1, C2, ωi)
    u0 = generate_u0(N)
    ds = CoupledODEs(sl_harmo_hetero!, u0, p0)
    X, t = trajectory(ds, T; Ttr, Δt)
    return(X, t)
end

function numerical_simulation_harmo(K::Float64, C1::Float64,C2::Float64, ω_dist::Distribution{Univariate,Continuous}, u0_given; Ttr::Int = 50, Δt::Float64 = 0.1, T::Int = 20)
    ωi = rand(ω_dist, N)
    p0 = (K, C1, C2, ωi)
    u0 = u0_given
    ds = CoupledODEs(sl_harmo_hetero!, u0, p0)
    X, t = trajectory(ds, T; Ttr, Δt)
    return(X, t)
end

function numerical_simulation_harmo(K::Float64, C1::Float64,C2::Float64, ω_dist::Vector{Float64}, u0_given; Ttr::Int = 50, Δt::Float64 = 0.1, T::Int = 20)
    ωi = ω_dist
    p0 = (K, C1, C2, ωi)
    u0 = u0_given
    ds = CoupledODEs(sl_harmo_hetero!, u0, p0)
    X, t = trajectory(ds, T; Ttr, Δt)
    return(X, t)
end

function KOP_s(amps::Vector{Float64}, phases::Vector{Float64})
    N = length(amps)
    sum_exp_A = sum(amps .* exp.(1im * phases))
    sum_exp = sum(exp.(1im * phases))
    return(abs(sum_exp_A)/N, abs(sum_exp)/N)
end

function mean_frequencies(X::StateSpaceSet, t;
                          primary  = :autocorrelation,
                          fallback = :periodogram)
    N = Int(size(X[1,:],1)÷2)
    freqs = zeros(Float64, N)
    for k in 1:N
        reals = X[:, 2k-1]
        period = try
            estimate_period(reals, primary, t)
        catch                 
            try
                estimate_period(reals, fallback, t)
            catch             
                NaN
            end
        end

        freqs[k] = isnan(period) ? NaN : 2π / period
    end
    return freqs
end

function instantaneous_frequencies(X::StateSpaceSet, Δt = Float64)
    N = Int(size(X[1,:],1)÷2)
    freqs = Vector{Vector{Float64}}(undef, N)  # vector of N vectors

    for i in 1:N
        reals = X[:, 2i - 1]
        imags = X[:, 2i]
        phases = angle.(reals .+ im .* imags)
        unwrapped_phases = unwrap(phases)
        freqs[i] = diff(unwrapped_phases) ./ Δt
    end
    return freqs
end