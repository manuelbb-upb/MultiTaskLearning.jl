import .COSMO
export COSMOConfig

struct COSMOConfig <: AbstractJuMPConfig
    cfg :: JuMPConfig
end

function COSMOConfig(;eps_abs::Real=1e-5, max_iter=10_000, kwargs...)
    attributes = Dict("eps_abs" => eps_abs, "max_iter" => max_iter)
    if haskey(kwargs, :attributes)
        merge!(attributes, kwargs[:attributes])
        delete!(kwargs, :attributes)
    end

    COSMOConfig(JuMPConfig(; solver=COSMO.Optimizer, attributes, kwargs...))
end

multidir(Df::AbstractMatrix, cfg::COSMOConfig) = multidir(Df, cfg.cfg)