using Pkg

# make stuff in `../exps`
const EXPS_PATH="\"$(joinpath(@__DIR__, "..", "exps"))\""
exps_preprocess = function (content)
    content = replace(content, "@__DIR__" => EXPS_PATH)
    return content
end

current_env = first(Base.load_path())
Pkg.activate(joinpath(@__DIR__, "..", "exps"))
Pkg.develop(PackageSpec(;path=joinpath(@__DIR__, "..")))
Pkg.instantiate()

Literate.markdown(
    joinpath(@__DIR__, "..", "exps", "multidir_descent.jl"), 
    joinpath(@__DIR__, "src", "exps"); 
    flavor=Literate.DocumenterFlavor(),
    preprocess=exps_preprocess,
    #postprocess=exps_postprocess,
)
Pkg.activate(current_env)

# make other scripts
Pkg.add("LinearAlgebra")
Literate.markdown(
    joinpath(@__DIR__, "..", "src", "multidir_frank_wolfe.jl"),
    joinpath(@__DIR__, "src"); 
    flavor=Literate.DocumenterFlavor()
)
