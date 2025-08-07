# GeneralizedChisqDistribution

This package implements the [Generalized Chi-squared distribution](https://en.wikipedia.org/wiki/Generalized_chi-squared_distribution) in Julia.

## Installation

Add this package to the current environment with `]add GeneralizedChisqDistribution` in the "pkg mode" of the REPL, or in the "standard REPL":

```julia
using Pkg
Pkg.add("GeneralizedChisqDistribution")
```

## Usage

`GeneralizedChisqDistribution` extends the functionality of [Distributions.jl](https://github.com/JuliaStats/Distributions.jl) by exporting the univariate continuous distribution `GeneralizedChisq`.

For example, calculate the CDF at `x = 5.0` of a distribution that represents
the weighted sum of two noncentral Chi-squared and a Normal variable with the following parameters:
* Weights (`w`) of the Chi-squared distributions equal to `1` and `-1`.
* Degrees of freedom (`ν`) `1` and `2`.
* Noncentrality parameters (`λ`) equal to `1.5` for both distibutions.
* Mean of the normal variable (`μ`) equal to `10`.
* Standard deviation of the normal variable (`σ`) equal to `1`.

```julia
using Distributions
using GeneralizedChisqDistribution

d = GeneralizedChisq([1,-1], [1,2], [1.5, 1.5], 10, 1)

cdf(d, 5.0)
```

## API

```@docs
GeneralizedChisq
```