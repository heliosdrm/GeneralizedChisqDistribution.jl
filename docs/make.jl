using GeneralizedChisqDistribution
using Documenter

makedocs(;
    authors="Helios De Rosario",
    sitename="GeneralizedChisqDistribution.jl",
    format=Documenter.HTML(;
        canonical="https://heliosdrm.github.io/GeneralizedChisqDistribution.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md"
    ],
)

deploydocs(;
    repo="github.com/heliosdrm/GeneralizedChisqDistribution.jl",
    devbranch="main",
)
