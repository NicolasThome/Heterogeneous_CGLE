using Roots

BF(C1::Float64, C2::Float64) = -2*(1+C1*C2)/(1+C1^2)

function SN_solitary(K::Float64, C1::Float64, C2::Float64)
    disc = -4*(C1-C2)^2*K^2 - 20*(2*C1+1)*K + 25
    disc < 0 && return NaN
    A = sqrt(disc)
    return 2*(A - 10)*(2C1 + 1)*K + 5*(A + 5) - 4*(C1 - 2)^2*K^2
end

function HB_solitary(K::Float64, C1::Float64, C2::Float64)
    disc = -4*(C1-C2)^2*K^2 - 20*(2C1+1)*K + 25
    disc < 0 && return NaN
    A = sqrt(disc)
    return A+(4*C1-3)*K
end

function solve_curve(f, C1_list;
                     C2::Float64 = 2.0,
                     K_min::Float64 = 0.0,
                     K_max::Float64 = 1.5,
                     K_steps::Int = 400,
                     method = Bisection())
    K_grid = range(K_min, K_max; length=K_steps)
    K_root = fill(NaN, length(C1_list))

    for (idx, C1) in pairs(C1_list)
        # evaluate f on the coarse K grid for this C1
        vals = map(K -> f(K, C1, C2), K_grid)

        # look for the first sign change
        found = false
        for j in 1:(K_steps-1)
            v1, v2 = vals[j], vals[j+1]
            if isfinite(v1) && isfinite(v2) && signbit(v1) != signbit(v2)
                # bracket found → refine
                try
                    K_root[idx] = find_zero(K -> f(K, C1, C2),
                                            (K_grid[j], K_grid[j+1]),
                                            method)
                    found = true
                catch
                    # refinement failed – keep NaN
                end
                break
            end
        end
        !found && (K_root[idx] = NaN)   # optional, already NaN by default
    end
    return K_root
end