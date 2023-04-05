using Documenter
using MultiTaskLearning

makedocs(
    sitename = "MultiTaskLearning",
    format = Documenter.HTML(ansicolor = true),
    modules = [MultiTaskLearning],
    pages = [
        "index.md",
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
