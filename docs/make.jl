using MNN
using Documenter

DocMeta.setdocmeta!(MNN, :DocTestSetup, :(using MNN); recursive=true)

makedocs(;
    modules=[MNN],
    authors="Alexander Reimer <alexander.reimer2357@gmail.com>, Matteo Friedrich <matteo.r.friedrich@gmail.com>",
    repo="https://github.com/alex/MNN.jl/blob/{commit}{path}#{line}",
    sitename="MNN.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://alex.github.io/MNN.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/alex/MNN.jl",
    devbranch="main",
)
