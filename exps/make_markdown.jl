using Pkg
Pkg.activate(@__DIR__)
import Literate

cfg = Dict();
cfg[:execute] = false
cfg[:flavor] = Literate.DocumenterFlavor()
cfg[:documenter] = false

Literate.markdown(
    joinpath(@__DIR__, "multidir_descent.jl"), joinpath(@__DIR__, "../docs/src/exps"); config=cfg)