module GeneralizedChisqDistribution

using Statistics, Distributions
import Random: AbstractRNG
import QuadGK: quadgk

import Distributions: quantile_newton, params, sampler,
                      cdf, pdf, logpdf, cf, insupport

import Base: eltype, rand
import Base: maximum, minimum

import Statistics: mean, quantile, var

export GeneralizedChisq

"""
    GeneralizedChisq(w, ν, λ, μ, σ)

The *Generalized chi-squared distribution* is the distribution of a sum of independent noncentral chi-squared variables and a normal variable:

```math
\\xi =\\sum_{i}w_{i}y_{i}+x,\\quad y_{i}\\sim \\chi^{2}(\\nu_{i},\\lambda _{i}),\\quad x\\sim N(\\mu,\\sigma^{2}).
```

```julia
GeneralizedChisq(w, ν, λ, μ, σ)

```

External links

* [Generalized chi-squared distribution on Wikipedia](https://en.wikipedia.org/wiki/Generalized_chi-squared_distribution)

"""
struct GeneralizedChisq{T<:Real, V<:AbstractVector{<:Real}} <: ContinuousUnivariateDistribution
    w::V
    ν::V
    λ::V
    μ::T
    σ::T

    function GeneralizedChisq{T, V}(w, ν, λ, μ, σ; check_args::Bool=true) where {T, V}
        # @check_args GeneralizedChisq (ν, all(ν .> 0)) (σ, σ ≥ zero(σ)) (length(w) == length(ν) == length(λ))
        new{T, V}(w, ν, λ, μ, σ)
    end
end

# Manage parameter types, ensuring that the parameters of normal are floats.

function GeneralizedChisq(w, ν, λ, μ::Tμ, σ::Tσ; check_args...) where {Tμ<:Real, Tσ<:Real}
    V = promote_type(typeof(w), typeof(ν), typeof(λ))
    T = promote_type(Tμ, Tσ, eltype(w), eltype(ν), eltype(λ))
    GeneralizedChisq{T, V}(w, ν, λ, μ, σ...; check_args...)
end

GeneralizedChisq(w, ν, λ, μ::Integer, σ::Integer; check_args...) = GeneralizedChisq(w, ν, λ, float(μ), float(σ); check_args...)

Base.eltype(::GeneralizedChisq{T,V}) where {T,V} = T

params(d::GeneralizedChisq) = (d.w, d.ν, d.λ, d.μ, d.σ)

"""
    GeneralizedChisqSampler

Sampler of a generalized chi-squared distribution,
created by `sampler(::GeneralizedChisq)`.
"""
struct GeneralizedChisqSampler{T<:Real, SC<:Sampleable{Univariate, Continuous}, SN<:Sampleable{Univariate, Continuous}} <: Sampleable{Univariate, Continuous}
    μ::T
    nchisqsamplers::Vector{Tuple{T, SC}}
    normalsampler::SN # (zero-mean, since μ is defined apart)
    skipnormal::Bool  # Normal needs not be sampled if σ == 0
end

## Required functions

# sampler that predefines the distributions for batch sampling
function sampler(d::GeneralizedChisq{T}) where T
    μ = d.μ
    nchisqsamplers = [(T(d.w[i]), sampler(NoncentralChisq(d.ν[i], d.λ[i]))) for i in eachindex(d.w)]
    normalsampler = sampler(Normal(zero(T), d.σ)) # zero-mean
    skipnormal = iszero(d.σ)
    GeneralizedChisqSampler(μ, nchisqsamplers, normalsampler, skipnormal)
end

function rand(rng::AbstractRNG, s::GeneralizedChisqSampler)
    result = s.μ
    for (w, ncs) in s.nchisqsamplers
        result += w * rand(rng, ncs)
    end
    if !s.skipnormal
        result += rand(rng, s.normalsampler)
    end
    return result
end

rand(rng::AbstractRNG, d::GeneralizedChisq) = rand(rng, sampler(d))

# cdf algorithm derived from https://github.com/abhranildas/gx2, by Abhranil Das.
function cdf(d::GeneralizedChisq{T}, x::Real) where T
    # special cases
    if isempty(d.w)
        return cdf(Normal(d.μ, d.σ), x)
    elseif iszero(d.σ)
        # out of support (or all weights equal to zero)
        (x < minimum(d)) && return zero(T)
        (x ≥ maximum(d)) && return one(T)
        # inside support
        if all(==(first(d.w)), d.w)
            nchisq = NoncentralChisq(sum(T, d.ν), sum(T, d.λ))
            (first(d.w) > zero(T)) && return cdf(nchisq, (x - d.μ)/first(d.w))
            (first(d.w) < zero(T)) && return ccdf(nchisq, (x - d.μ)/first(d.w))
        end
    end
    # general case
    (x == -Inf) && return zero(T)
    (x == Inf) && return one(T)
    GChisqComputations.daviescdf(d, x)
end

function pdf(d::GeneralizedChisq{T}, x::Real) where T
    # special cases
    if isempty(d.w)
        return pdf(Normal(d.μ, d.σ), x)
    elseif iszero(d.σ)
        # out of support
        !insupport(d, x) && return zero(T)
        # inside support
        if all(==(first(d.w)), d.w)
            iszero(first(d.w)) && return typemax(T) # zero weights (delta at d.μ)
            nchisq = NoncentralChisq(sum(T, d.ν), sum(T, d.λ))
            return pdf(nchisq, (x - d.μ)/first(d.w)) / abs(first(d.w))
        end
    end
    # general case
    !isfinite(x) && return zero(T)
    GChisqComputations.daviespdf(d, x)
end

logpdf(d::GeneralizedChisq, x::Real) = log(pdf(d, x))

function quantile(d::GeneralizedChisq{T}, p::Real) where T
    if 0 < p < 1
        # special cases
        if isempty(d.w)
            return quantile(Normal(d.μ, d.σ), x)
        elseif iszero(d.σ) && all(==(first(d.w)), d.w)
            iszero(first(d.w)) && return d.μ # zero weights (delta at d.μ)
            nchisq = NoncentralChisq(sum(T, d.ν), sum(T, d.λ))
            (first(d.w) > zero(T)) && return quantile(nchisq, p)*first(d.w) + d.μ
            (first(d.w) < zero(T)) && return cquantile(nchisq, p)*first(d.w) + d.μ
        end
        # general cases
        return GChisqComputations.quantile(d, p)
    elseif p == 0
        return minimum(d)
    elseif p == 1
        return maximum(d)
    else
        return T(NaN)
    end
end

function minimum(d::GeneralizedChisq{T}) where T
    d.σ > zero(T) || any(<(zero(T)), d.w) ? typemin(T) : d.μ
end

function maximum(d::GeneralizedChisq{T}) where T
    d.σ > zero(T) || any(>(zero(T)), d.w) ? typemax(T) : d.μ
end

insupport(d::GeneralizedChisq, x::Real) = minimum(d) ≤ x ≤ maximum(d)

## Recommended functions

mean(d::GeneralizedChisq) = d.μ + sum(d.w[i] * (d.ν[i] + d.λ[i]) for i in eachindex(d.w))
var(d::GeneralizedChisq) = d.σ^2 + 2 * sum(d.w[i]^2 * (d.ν[i] + 2*d.λ[i]) for i in eachindex(d.w))
cf(d::GeneralizedChisq, t) = GChisqComputations.cf_explicit(d, t)

# # [Missing]
# modes(d::GeneralizedChisq)
# mode(d::GeneralizedChisq)
# skewness(d::GeneralizedChisq)
# kurtosis(d::GeneralizedChisq, ::Bool)
# entropy(d::GeneralizedChisq, ::Real)
# mgf(d::GeneralizedChisq, ::Any)

include("GChisqComputations.jl")

end # module GeneralizedChisqDistribution
