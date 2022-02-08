using BpAlignGpu
using Documenter

DocMeta.setdocmeta!(BpAlignGpu, :DocTestSetup, :(using BpAlignGpu); recursive=true)

makedocs(;
    modules=[BpAlignGpu],
    authors="Andrea Pagnani, Politecnico di Torino",
    repo="https://github.com/pagnani/BpAlignGpu.jl/blob/{commit}{path}#{line}",
    sitename="BpAlignGpu.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://pagnani.github.io/BpAlignGpu.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/pagnani/BpAlignGpu.jl",
    devbranch="main",
)
