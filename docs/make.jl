using Documenter
using MultiTaskLearning

makedocs(
    sitename = "MultiTaskLearning",
    format = Documenter.HTML(),
    modules = [MultiTaskLearning]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "https://github.com/manuelbb-upb/MultiTaskLearning.jl",
    versions=nothing
)
