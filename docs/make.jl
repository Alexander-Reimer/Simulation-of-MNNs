using MNN
using Documenter

DocMeta.setdocmeta!(MNN, :DocTestSetup, :(using MNN); recursive=true)

makedocs(;
    modules=[MNN],
    authors="Alexander Reimer <alexander.reimer2357@gmail.com>, Matteo Friedrich <matteo.r.friedrich@gmail.com>",
    repo="https://github.com/Alexander-Reimer/Simulation-of-MNNs/blob/{commit}{path}#{line}",
    sitename="MNN.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://Alexander-Reimer.github.io/Simulation-of-MNNs",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Alexander-Reimer/Simulation-of-MNNs",
    devbranch="main",
)
