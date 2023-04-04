import .COSMO
export COSMOConfig

struct COSMOConfig <: AbstractJuMPConfig
    cfg :: JuMPConfig
end

function COSMOConfig(;eps_abs::Real=1e-5, max_iter=10_000, kwargs...)
    attributes = Dict{String, Any}("eps_abs" => eps_abs, "max_iter" => max_iter)
    if haskey(kwargs, :attributes)
        merge!(attributes, kwargs[:attributes])
        delete!(kwargs, :attributes)
    end

    COSMOConfig(JuMPConfig(; solver=COSMO.Optimizer, attributes, kwargs...))
end

_multidir(grads, cfg::COSMOConfig) = _multidir(grads, cfg.cfg)