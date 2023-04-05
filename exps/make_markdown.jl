using Pkg
Pkg.activate(@__DIR__)
import Literate

cfg = Dict();
cfg[:execute] = true
cfg[:flavor] = Literate.DocumenterFlavor()
cfg[:documenter] = false
cfg[:preprocess] = function set_env(content)
    content = replace(content, "@__DIR__" => "\"$(@__DIR__)\"")
    return content
end

Literate.markdown(
    joinpath(@__DIR__, "multidir_descent.jl"), joinpath(@__DIR__, "../docs/src/exps"); config=cfg)