using SLH
using LsqFit
using Statistics, LinearAlgebra
softplus(x, β) = log1p(exp(β*x)) / β    

@inline function Ω_model_fixed(K, p; Ω∞=-2.0)
    m, Kc, β = p
    return Ω∞ .+ m .* softplus.(Kc .- K, β)
end

nearest_index(v::AbstractVector{<:Real}, x::Real) =
    argmin(abs.(v .- x))

nearest(v::AbstractVector{<:Real}, x::Real) = begin
    i = nearest_index(v, x)
    return i, v[i]
end


