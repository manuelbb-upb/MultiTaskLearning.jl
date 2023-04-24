using Pkg
current_env = first(Base.load_path())
Pkg.activate(@__DIR__)

try
    using Documenter
    using Literate
    using MultiTaskLearning

    include("make_literate.jl")

    makedocs(
        sitename = "MultiTaskLearning",
        format = Documenter.HTML(;
            ansicolor = true,
            mathengine = Documenter.MathJax3(),
        ),
        modules = [MultiTaskLearning],
        pages = [
            "index.md",
            "multidir_frank_wolfe.md", 
            "exps/multidir_descent.md"
        ]
    )

    # Documenter can also automatically deploy documentation to gh-pages.
    # See "Hosting Documentation" and deploydocs() in the Documenter manual
    # for more information.
    deploydocs(
        repo = "github.com/manuelbb-upb/MultiTaskLearning.jl.git",
        versions=nothing
    )
catch err
    println("ERROR: ", err)
    stacktrace(catch_backtrace())
    
    Pkg.activate(current_env)
end