using Distributions, Roots, KernelDensity, FiniteDiff

function predict_radii_distribution(c_dist::Distribution{Univariate,Continuous}, K::Float64, C1 :: Float64, R::Float64, Ω::Float64, N :: Int)

    C2_samples = rand(c_dist, N)
    r_samples = zeros(N)
    
    # Coefficients α and β
    α = (K - 1) + C1 * (Ω + K * C1)
    β = Ω + K * C1 - C1 * (K - 1)
    T = (K * R * (1 + C1^2))^2
    
    for i in 1:N
        c_val = C2_samples[i]
        P = (1 + C1 * c_val)^2 + (c_val - C1)^2
        Q = 2 * (1 + C1 * c_val) * α + 2 * (c_val - C1) * β
        S = α^2 + β^2
        
        # Solve cubic equation for z = r^2
        f(z) = P * z^3 + Q * z^2 + S * z - T
        roots = find_zeros(f, 0.0, 10.0)
        positive_roots = roots[roots .> 0]
        
        if !isempty(positive_roots)
            z = minimum(positive_roots) 
            r_samples[i] = sqrt(z)
        else
            r_samples[i] = NaN  
        end
    end
    
    r_samples = r_samples[.!isnan.(r_samples)]
    
    return r_samples
end

function local_maxima(ys::AbstractVector{<:Real}, w::Int=10)
    N = length(ys)
    idxs = Int[]
    @inbounds for i in (w+1):(N-w)
        yi = ys[i]
        # check if yi exceeds its w neighbours on both sides
        if yi > maximum(ys[i-w:i-1]) && yi > maximum(ys[i+1:i+w]) && yi > 0.05
            push!(idxs, i)
        end
    end
    return idxs
end

function get_rho_distribution(r_sample::Vector{Float64}, w::Int, two_highest::Bool)

    kde_est = kde(r_sample)
    xs, ys  = kde_est.x, kde_est.density

    max_idx = local_maxima(ys, w)
    nmax = length(max_idx)

    mlocs_all = xs[max_idx]
    mdens_all = ys[max_idx]

    dx = xs[2] - xs[1] 
    d2ys = similar(ys)
    d2ys[1] = 0.0
    d2ys[end] = 0.0
    for i in 2:length(ys)-1
        d2ys[i] = (ys[i+1] - 2*ys[i] + ys[i-1]) / dx^2
    end
    curvatures_all = d2ys[max_idx]

    if nmax == 1
        return 0.0, mlocs_all, mdens_all, NaN

    elseif nmax == 2
        Δ = abs(mlocs_all[1] - mlocs_all[2])
        mean_width = sqrt(abs(1/curvatures_all[1]) + abs(1/curvatures_all[2]))
        separation_ratio = Δ / mean_width

        # thresh = mean(mlocs_all)
        # prop   = count(x -> x > thresh, r_sample) / length(r_sample)
        idx = sortperm(mdens_all; rev = true)
        amp1 = count(x -> abs(x - mlocs_all[idx[1]]) < 0.15, r_sample)  # tallest
        amp2 = count(x -> abs(x - mlocs_all[idx[2]]) < 0.15, r_sample)  # second-tallest
        prop = amp1/(amp2+amp1)
        return prop, mlocs_all, mdens_all, separation_ratio

    elseif nmax > 2 && two_highest
        sorted = sortperm(mdens_all, rev = true)[1:2]
        mlocs = mlocs_all[sorted]
        mdens = mdens_all[sorted]
        curvatures = curvatures_all[sorted]

        Δ = abs(mlocs[1] - mlocs[2])
        mean_width = sqrt(abs(1/curvatures[1]) + abs(1/curvatures[2]))
        separation_ratio = Δ / mean_width

        thresh = mean(mlocs)
        prop   = count(x -> x > thresh, r_sample) / length(r_sample)

        return prop, mlocs, mdens, separation_ratio

    else
        return NaN, mlocs_all, mdens_all, NaN
    end
end


