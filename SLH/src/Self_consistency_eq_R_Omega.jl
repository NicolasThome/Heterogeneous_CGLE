using Distributions, Roots, HCubature, NLsolve

function self_consistency(x,
                          c_dist::Distribution{Univariate,Continuous},
                          K::Float64,
                          C1::Float64)
    R, Ω = x
    σ = std(c_dist)
    C2_mean = mean(c_dist)
    low = C2_mean - 10*σ
    high = C2_mean + 10*σ

    function integrand(c)

        c_val = c[1]
        

        α = (K - 1) + C1*(Ω + K*C1)
        β = (Ω + K*C1) - C1*(K - 1)
        P = (1 + C1*c_val)^2 + (c_val - C1)^2
        Q = 2*(1 + C1*c_val)*α + 2*(c_val - C1)*β
        S = α^2 + β^2
        T = (K * R * (1 + C1^2))^2
        
        # Cubic: P*z^3 + Q*z^2 + S*z - T = 0
        f(z) = P*z^3 + Q*z^2 + S*z - T
        roots = find_zeros(f, 0.0, 10.0)  # Find real roots in [0,10]
        
        # Select smallest positive real root
        positive_roots = roots[roots .> 0]
        if isempty(positive_roots)
            return [0.0, 0.0]  # Skip points with no solution
        end
        z = maximum(positive_roots)
        r = sqrt(z)
        
        # Compute trigonometric components
        A_val = (1 + C1*c_val)*z + α
        B_val = (c_val - C1)*z + β
        denom = K * R * (1 + C1^2)
        cosψ = (r / denom) * A_val
        sinψ = (r / denom) * B_val
        
        # Weight by Gaussian PDF
        pdf_val = pdf(c_dist, c_val)
        return [r * cosψ * pdf_val, r * sinψ * pdf_val]
    end
    
    # Perform vector-valued integration
    result, _ = hcubature(integrand, [low], [high], rtol=1e-4)
    return [result[1] - R, result[2]]  # Residuals: [⟨r cosψ⟩ - R, ⟨r sinψ⟩]
end

function find_phase_locked_state(current_guess::AbstractVector{<:Real},
                           c_dist,
                           K::Float64,
                           C1::Float64;
                           method=:newton,
                           autodiff=:forward,
                           ftol=1e-6,
                           maxiters=500,
                           show_trace=true,
                           store_trace=true,
                           extended_trace=true)

    sol = nlsolve(x -> self_consistency(x, c_dist, K, C1),
                  current_guess;
                  method         = method,
                  autodiff       = autodiff,
                  ftol           = ftol,
                  iterations     = maxiters,
                  show_trace     = show_trace,
                  store_trace    = store_trace,
                  extended_trace = extended_trace)

    return sol.zero               # a 2‐vector [R_sol, Ω_sol]
end 