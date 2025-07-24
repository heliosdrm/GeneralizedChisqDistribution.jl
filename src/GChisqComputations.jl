"""
    GChisqComputations

Computations for the Generalized Chi-squared distribution.
This module is meant to be private, not part of the API.
"""
module GChisqComputations
    import ..Distributions # to use `Normal` and `NoncentralChisq` (which cannot be imported)
    import ..GeneralizedChisq
    import ..cf, ..cdf, ..insupport, ..quantile_newton
    import ..quadgk, ..mean, ..var

    # Characteristic function - explicit formula
    function cf_explicit(d, t)
        arg = im * d.μ * t
        denom = Complex(one(d.μ) * one(t)) # unit with stable type in later operations
        for i in eachindex(d.w)
            arg += im * d.w[i] * d.λ[i] * t / (1 - 2im * d.w[i] * t)
            denom *= (1 - 2im * d.w[i] * t)^(d.ν[i]/2)
        end
        arg -= d.σ^2 * t^2 / 2
        return exp(arg) / denom
    end

    # Characteristic function - by inheritance
    function cf_inherit(d::GeneralizedChisq, t)
        result = exp(im * d.μ * t)
        for i in eachindex(d.w)
            result *= cf(Distributions.NoncentralChisq(d.ν[i], d.λ[i]), d.w[i] * t)
        end
        if !iszero(d.σ)
            result *= cf(Distributions.Normal(zero(d.μ), d.σ), t)
        end
        return result
    end

    """
        daviesterms(d::GeneralizedChisq, u, x)

    Terms of the formula (13) in Davies (1980) to calculate
    the cdf of a generalized chi-squared distribution, as: 
        
        F(x) = 1/2 - 1/π *∫sin(θ)/(u*ρ)du

    where `θ` and `ρ` are the outputs of this function.

    Those terms are related to the characteristic function of the distribution as:
        
        exp(θ*im)/ρ = exp(-u*x*im)*cf(u) 

    They can be also used to calculate the pdf as:
        
        f(x) = 1/π *∫cos(θ)/ρ du

    And its derivative as:
        
        f'(x) = 1/π *∫u*sin(θ)/ρ du
    
    Reference:

    Robert B. Davies (1980).
    Algorithm AS 155: The Distribution of a Linear Combination of χ2Random Variables.
    *Journal of the Royal Statistical Society. Series C (Applied Statistics)*, 29(3), 323–333
    DOI:10.2307/2346911  
    """
    function daviesterms(d::GeneralizedChisq{T}, u, x) where {T<:Real}
        θ = -T(u*(x - d.μ))
        ρ = exp(T(u*d.σ)^2 / 2)
        for i in eachindex(d.w)
            wi, νi, λi = T(d.w[i]), T(d.ν[i]), T(d.λ[i])
            τ  = 1 + 4 * wi^2 * u^2
            θ += νi*atan(2*wi*u)/2 + (λi*wi*u)/τ
            ρ *= τ^(νi/4) * exp(2λi*wi^2*u^2/τ)
        end
        return θ, ρ
	end

    function daviescdf(d, x)
        atol = eps(eltype(d)) # uniform tolerance to avoid issues at cdf ≈ 0.5 (integral ≈ 0)
        integral, _ = quadgk(0, Inf; atol=atol) do u
            θ, ρ = daviesterms(d, u, x)
            return sin(θ)/(u*ρ)
        end
        return 1/2 - integral/π
    end

    function daviespdf(d, x)
        integral, _ = quadgk(0, Inf) do u
            θ, ρ = daviesterms(d, u, x)
            return cos(θ)/ρ
        end
        return integral/π
    end

    # derivative of pdf
    function daviesdpdf(d, x)
        integral, _ = quadgk(0, Inf) do u
            θ, ρ = daviesterms(d, u, x)
            return u*sin(θ)/ρ
        end
        return integral/π
    end

    # functions to look for the starting point where
    # the Newton method for quantiles converges:
    # sign(F(x) - q) == sign(f'(x))

    # Convergence criterion:
    function newtonconvergence(d::GeneralizedChisq{T}, p, x) where T
        error = cdf(d, x) - T(p)
        # out of bounds
        iszero(d.σ) && !insupport(d, x) && return error, zero(T), false
        # general case
        curvature = daviesdpdf(d, x)
        converges = sign(error) == sign(curvature)
        return error, curvature, converges
    end

    # Bracket containing solution for q, with x as one of the ends
    function definebracket(d::GeneralizedChisq, p, x, errorx, curvx, initialwidth)
        # set direction of span and initialize bracket
        span = (errorx < 0) ? abs(initialwidth) : -abs(initialwidth)
        xbis = x + span
        errorxbis, curvxbis, _ = newtonconvergence(d, p, xbis)
        # greedier search if the bracket does not contain the solution
        while sign(errorxbis) == sign(errorx)
            span *= 2*errorxbis / (errorx - errorxbis)
            x, errorx, curvx = xbis, errorxbis, curvxbis
            xbis = x + span
            errorxbis, curvxbis, _ = newtonconvergence(d, p, xbis)
        end
        # bracket end out of bounds
        if !insupport(d, xbis)
            xbis = d.μ
        end
        return (x, xbis), (errorx, errorxbis), (curvx, curvxbis)
    end

    # Bisect bracket until convergence point is reached
    function searchnewtonconvergence(d, p, (x,xbis), (errorx, errorxbis), (curvx, curvxbis))
        while true
            # Assumes that the first point is *outside* the region of convergence
            if sign(errorxbis) == sign(curvxbis)
                return xbis
            end
            xmid = (x + xbis) / 2
            errorxmid, curvxmid, _ = newtonconvergence(d, p, xmid)
            # switch first point if the new end of the bracket is on the same side
            if sign(errorxmid) == sign(errorx)
                x, errorx, curvx = xbis, errorxbis, curvxbis
            end
            xbis, errorxbis, curvxbis = xmid, errorxmid, curvxmid
        end
    end

    function quantile(d, p)
        # search starting point meeting convergence criterion
        x0 = mean(d)
        error0, curv0, converges = GChisqComputations.newtonconvergence(d, p, x0)
        if !converges
            bracket = GChisqComputations.definebracket(d, p, x0, error0, curv0, sqrt(var(d)))
            x0 = GChisqComputations.searchnewtonconvergence(d, p, bracket...)
        end
        return quantile_newton(d, p, x0)
    end
end
