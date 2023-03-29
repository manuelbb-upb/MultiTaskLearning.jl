
import .JuMP: Model, @variable, @constraint, @objective, @expression, optimize!, value,
    set_silent, set_optimizer_attribute, MOI
export JuMPConfig

abstract type AbstractJuMPConfig<:AbstractMultiDirConfig end

Base.@kwdef struct JuMPConfig{S<:MOI.AbstractOptimizer} <: AbstractJuMPConfig
    solver :: S
    target :: Symbol = :dual
    verbose :: Bool = false
    attributes :: Dict{String, Any} = Dict{String, Any}()
end

function setup_jump_problem(cfg :: JuMPConfig)
    opt = Model(cfg.solver)
    if !cfg.verbose
        set_silent(opt)
    end
    for (attr_str, attr) in cfg.attributes
        set_optimizer_attribute(opt, attr_str, attr)
    end
    return opt
end

function multidir(Df::AbstractMatrix, cfg::JuMPConfig)
    opt = setup_jump_problem(cfg)
    if cfg.target == :dual
        return jump_multidir_dual(opt, Df)
    elseif cfg.target == :primal
        return jump_multidir_primal(opt, Df)
    else
        @error "Unknown target `$(cfg.target)`. Must be `:primal` or `:dual`."
    end
end

@doc raw"""
Return a convex combination of negative gradients by solving the dual problem
```math
\min_{α ∈ ℝᴷ, α >= 0} ‖ -Dfᵀ ⋅ α ‖²
```
"""
function jump_multidir_dual(opt, Df)
    num_objfs, num_vars = size(Df)

    @variable(opt, α[1:num_objfs] .>= 0)
    @expression(opt, d, -Df'α)
    @objective(opt, Min, sum(d.^2))
    @constraint(opt, sum(α) == 1)

    optimize!(opt)
    return value.(d)
end

@doc raw"""
Return the minimizer of the primal problem
```math
\min_{d ∈ ℝⁿ} \maxₖ (Df ⋅ d)ₖ + ½ ‖d‖²
```
"""
function jump_multidir_primal(opt, Df)
    num_objfs, num_vars = size(Df)

    @variable(opt, β)
    @variable(opt, d[1:num_vars])

    @objective(opt, Min, β + 0.5 * sum(d.^2))
    @constraint(opt, Df*d .<= β)

    optimize!(opt)

    return value.(d)
end