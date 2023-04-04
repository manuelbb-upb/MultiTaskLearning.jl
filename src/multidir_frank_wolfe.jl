import LinearAlgebra as LA
#=
## Custom Frank-Wolfe Solver for Dual

Suppose ``v ∈ ℝⁿ`` and ``u ∈ ℝⁿ`` are vectors and ``M ∈ ℝ^{n×n}`` is a **symmetric** 
square matrix. What is the minimum of 
```math
σ(γ) = ( (1-γ) v + γ u )ᵀ M ( (1-γ) v + γ u) ?   (γ ∈ [0,1])
```

We have
```math
σ(γ) \begin{aligned}[t]
    &=
    ( (1-γ) v + γ u )ᵀ M ( (1-γ) v + γ u )
        \\
    &=
    (1-γ)² \underbrace{vᵀ M v}_{a} + 2γ(1-γ) \underbrace{uᵀ M v}_{b} + γ² \underbrace{uᵀ M u}_{c}
        \\
    &=
    (1 + γ² - 2γ)a + (2γ - 2γ²)b + γ² c 
        \\
    &=
    (a -2b + c) γ² + 2 (b-a) γ + a
\end{aligned}
```
The boundary values are
```math
σ₀ = σ(0) = a \text{and} σ₁ = σ(1) = c.
```
If ``(a-2b+c) > 0 ⇔ a-b > b-c``, 
then the parabola is convex and has its global minimum where the derivative is zero:
```math
2(a - 2b + c) y^* + 2(b-a) \stackrel{!}= 0 
 ⇔ 
    γ^* = \frac{-2(b-a)}{2(a -2 b + c)} 
        = \frac{a-b}{(a-b)+(c-b)}
```
If ``a-b < b -c``, the parabola is concave and this is a maximum.
The extremal value is 
```math
σ_* = σ(γ^*) 
    = \frac{(a - b)^2}{(a-b)+(c-b)} - \frac{2(a-b)^2}{(a-b) + (c-b)} + a
    = a - \frac{(a-b)^2}{(a-b) + (c-b)}
```
=#

"""
    min_quad(a,b,c)

Given a quadratic function ``(a -2b + c) γ² + 2 (b-a) γ + a`` with ``γ ∈ [0,1]``, return 
`γ_opt` minimizing the function in that interval and its optimal value `σ_opt`.
"""
function min_quad(a,b,c)
    a_min_b = a-b
    b_min_c = b-c
    if a_min_b > b_min_c
        ## the function is a convex parabola and has its global minimum at `γ`
        γ = a_min_b /(a_min_b - b_min_c)
        if 0 < γ < 1
            # if its in the interval, return it
            σ = a - a_min_b * γ
            return γ, σ
        end
    end
    ## the function is either a line or a concave parabola, the minimum is attained at the 
    ## boundaries
    if a <= c
        return 0, a
    else
        return 1, c
    end
end

function min_chull2(M, v, u)
    Mv = M*v
    a = v'Mv
    b = u'Mv
    c = u'M*u
    return min_quad(a,b,c)
end

function frank_wolfe_multidir_dual(
    grads#::AbstractVector{<:AbstractVector}
    ; 
    max_iter=10_000, eps_abs=1e-5
)
    num_objfs = length(grads)
    T = Base.promote_type(Float32, mapreduce(eltype, promote_type, grads))
    
    ## 1) Initialize ``α`` vector. There are smarter ways to do this...
    α = fill(T(1/num_objfs), num_objfs)

    ## 2) Build symmetric matrix of gradient-gradient products
    ### `_M` will be a temporary, upper triangular matrix
	_M = zeros(T, num_objfs, num_objfs)
	for (i,gi) = enumerate(grads)
		for (j, gj) = enumerate(grads)
            j<i && continue
			_M[i,j] = gi'gj
		end
	end
    ### mirror `_M` to get the full symmetric matrix
	M = LA.Symmetric(_M, :U)

    ## 3) Solver iteration
    _α = copy(α)    # to keep track of change
    for _=1:max_iter
        t = argmin( M*α )
        v = α
        u = zeros(T, num_objfs)
        u[t] = one(T)
        
        γ, _ = min_chull2(M, v, u)

        α .*= (1-γ)
        α[t] += γ

        if sum( abs.( _α .- α ) ) <= eps_abs
            break
        end
        _α .= α
    end
    
    #return -sum(α .* grads) # somehow, broadcasting leads to type instability here, see also https://discourse.julialang.org/t/type-stability-issues-when-broadcasting/92715
    return mapreduce(*, sum, α, grads)
end

Base.@kwdef struct FWConfig <: AbstractMultiDirConfig
    max_iter :: Int = 10_000
    eps_abs :: Float64 = 1e-5
end

function _multidir(grads, cfg::FWConfig)
    return frank_wolfe_multidir_dual(grads; max_iter=cfg.max_iter, eps_abs=cfg.eps_abs)
end