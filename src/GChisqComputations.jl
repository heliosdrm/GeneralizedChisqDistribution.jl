module GChisqComputations
import ..Distributions
import ..GeneralizedChisq
import ..cf, ..cdf, ..insupport
import ..quadgk, ..mean, ..var
import StaticArrays: SVector

# --- 1. Characteristic function -----------------------------------------------

function cf_explicit(d, t)
    arg = im * d.μ * t
    denom = Complex(one(d.μ) * one(t))
    for i in eachindex(d.w)
        arg += im * d.w[i] * d.λ[i] * t / (1 - 2im * d.w[i] * t)
        denom *= (1 - 2im * d.w[i] * t)^(d.ν[i] / 2)
    end
    arg -= d.σ^2 * t^2 / 2
    return exp(arg) / denom
end

# --- 2. Davies inversion terms ------------------------------------------------

function daviesterms(d::GeneralizedChisq{T}, u, x) where {T<:Real}
    θ = -T(u * (x - d.μ))
    ρ = exp(T(u * d.σ)^2 / 2)
    for i in eachindex(d.w)
        wi, νi, λi = T(d.w[i]), T(d.ν[i]), T(d.λ[i])
        τ = 1 + 4 * wi^2 * u^2
        θ += νi * atan(2 * wi * u) / 2 + (λi * wi * u) / τ
        ρ *= τ^(νi / 4) * exp(2λi * wi^2 * u^2 / τ)
    end
    return θ, ρ
end

# --- 3. Truncation Logic (Ported/Adapted from davies.cpp) ---------------------

function davies_truncation_err(u, d::GeneralizedChisq{T}, tausq) where T
    sum1 = zero(T)
    prod2 = zero(T)
    prod3 = zero(T)
    s = zero(T) # Keep as T to support non-integer degrees of freedom
    sigsq = T(d.σ^2)
    sum2 = (sigsq + tausq) * u^2
    prod1 = 2.0 * sum2
    u_scaled = 2.0 * u
    for i in eachindex(d.w)
        wi, ni, nci = T(d.w[i]), T(d.ν[i]), T(d.λ[i])
        x = (u_scaled * wi)^2
        sum1 += nci * x / (1.0 + x)
        if x > 1.0
            prod2 += ni * log(x)
            prod3 += ni * log(1.0 + x)
            s += ni
        else
            prod1 += ni * log(1.0 + x)
        end
    end
    sum1 *= 0.5
    prod2 = prod1 + prod2
    prod3 = prod1 + prod3
    x_val = exp(-sum1 - 0.25 * prod2) / π
    y_val = exp(-sum1 - 0.25 * prod3) / π

    err1 = (s == 0) ? 1.0 : x_val * 2.0 / s
    err2 = (prod3 > 1.0) ? 2.5 * y_val : 1.0
    res_err1 = min(err1, err2)
    res_err2 = (0.5 * sum2 <= y_val) ? 1.0 : y_val / (0.5 * sum2)
    return min(res_err1, res_err2)
end

function find_u_limit(d::GeneralizedChisq{T}, acc) where T
    sigsq = T(d.σ^2)
    # Variance-based heuristic for initial guess
    sd = sqrt(sigsq + sum(T(d.w[i])^2 * (2 * T(d.ν[i]) + 4 * T(d.λ[i])) for i in eachindex(d.w)))
    ut = 16.0 / sd
    u_guess = ut / 4.0

    if davies_truncation_err(u_guess, d, 0.0) > acc
        while davies_truncation_err(ut, d, 0.0) > acc
            ut *= 4.0
            ut > 1e6 && break
        end
    else
        ut = u_guess
        u_guess /= 4.0
        while davies_truncation_err(u_guess, d, 0.0) <= acc
            ut = u_guess
            u_guess /= 4.0
        end
    end
    # Refine the limit
    for divisor in (2.0, 1.4, 1.2, 1.1)
        u_test = ut / divisor
        if davies_truncation_err(u_test, d, 0.0) <= acc
            ut = u_test
        end
    end
    return ut
end

# --- 4. CDF and PDF Core ------------------------------------------------------

function daviescdf_pdf(d, x, u_lim; atol=1e-10, rtol=1e-5)
    (int_cdf, int_pdf), _ = quadgk(0, u_lim; atol=atol, rtol=rtol) do u
        θ, ρ = daviesterms(d, u, x)
        # Vectorized return for efficiency
        return SVector(sin(θ) / (u * ρ), cos(θ) / ρ)
    end
    return 1 / 2 - int_cdf / π, int_pdf / π
end

function daviescdf(d, x; acc=1e-6, atol=1e-8, rtol=1e-5)
    u_lim = find_u_limit(d, acc)
    return daviescdf_pdf(d, x, u_lim; atol=atol, rtol=rtol)[1]
end

function daviespdf(d, x; acc=1e-6, atol=1e-8, rtol=1e-5)
    u_lim = find_u_limit(d, acc)
    return daviescdf_pdf(d, x, u_lim; atol=atol, rtol=rtol)[2]
end

# --- 5. Robust Quantile (Hybrid Newton-Bisection) -----------------------------

function quantile(d::GeneralizedChisq{T}, p;
    acc=1e-5,
    atol=1e-8,
    rtol=1e-5,
    n_tol=1e-7
) where T
    u_lim = find_u_limit(d, acc)
    get_cp(val) = daviescdf_pdf(d, val, u_lim; atol=atol, rtol=rtol)

    # A. Initial Bracketing
    x = mean(d)
    c_val, p_val = get_cp(x)
    step = sqrt(var(d))
    low, high = x, x

    if c_val > p
        # Solution is to the left
        while true
            low -= step
            c_low, _ = get_cp(low)
            if c_low <= p
                high = low + step # The previous 'low' is a valid 'high'
                break
            end
            step *= 2
        end
    else
        # Solution is to the right
        while true
            high += step
            c_high, _ = get_cp(high)
            if c_high >= p
                low = high - step # The previous 'high' is a valid 'low'
                break
            end
            step *= 2
        end
    end

    # B. Safe Newton Loop (Hybrid with Bisection)
    # Start loop at the best guess (midpoint of the new bracket)
    x = (low + high) / 2
    for i in 1:60
        c_val, p_val = get_cp(x)
        err = c_val - T(p)

        if abs(err) < n_tol || (high - low) < n_tol * max(1, abs(x))
            return x
        end

        # Update bracket
        if err < 0
            low = x
        else
            high = x
        end

        # Newton attempt
        x_new = x - err / p_val

        # Safety: check bracket bounds and avoid vanishing gradients
        if x_new <= low || x_new >= high || p_val < 1e-12
            x = (low + high) / 2
        else
            x = x_new
        end
    end
    return x
end

end # module
