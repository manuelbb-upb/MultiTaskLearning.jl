```@meta
EditURL = "<unknown>/../src/multidir_frank_wolfe.jl"
```

# Custom Frank-Wolfe Solver...
... to compute the multi-objective steepest descent direction cheaply.
For unconstrained problems, the direction can be computed by projecting
``\symbf{0}\in ℝ^n`` onto the convex hull of the negative objective gradients.
This can be done easily with `JuMP` and a suitable solver (e.g., `COSMO`).

In “Multi-Task Learning as Multi-Objective Optimization” by Sener & Koltun, the authors
employ a variant of the Frank-Wolfe-type algorithms defined in
“Revisiting Frank-Wolfe: Projection-Free Sparse Convex Optimization” by Jaggi.

The objective for the projection problem is
```math
F(\symbf{α})
= \frac{1}{2} ‖ \sum_{i=1}^K αᵢ ∇fᵢ ‖_2^2
= \frac{1}{2} ‖ \nabla \symbf{f}^T \symbf{α} ‖_2^2
= \frac{1}{2} \symbf a^T \nabla \symbf{f} \nabla \symbf{f}^T \symbf {α}
```
Hence,
```math
\nabla F(\symbf{α})
= \nabla \symbf{f} \nabla \symbf{f}^T \symbf α
=: \symbf M \symbf α
```
The algorithm starts with some initial ``\symbf{α} = [α_1, …, α_K]^T``
and optimizes ``F`` within the standard simplex
``S = \{\symbf α = [α_1, …, α_k]: α_i \ge 0, \sum_i α_i = 1\}.``
This leads to the following procedure:

1) Compute seed ``s`` as the minimizer of
   ``\langle \symbf s, \nabla F(\symbf α_k) \rangle =
   \langle \symbf s, \symbf M \symbf α_k\rangle``
   in ``S``.
   The minimum is attained in one of the corners, i.e.,
   ``\symbf s = \symbf e_t``, where ``t`` is the minimizing index for the entries of ``\symbf M \symbf α_k``.
2) Compute the exact stepsize ``γ\in[0,1]`` that minimizes
   ```math
   F((1-γ)\symbf α_k + γ \symbf s).
   ```
3) Set ``\symbf α_{k+1} = (1-γ_k) α_k + γ_k \symbf s``.

## Finding the Stepsize

Let's discuss step 2.
Luckily, we can indeed (easily) compute the minimizing stepsize.
Suppose ``\symbf v ∈ ℝⁿ`` and ``\symbf u ∈ ℝⁿ`` are vectors and
``\symbf M ∈ ℝ^{n×n}`` is a **symmetric**
square matrix. What is the minimum of the following function?
```math
σ(γ) = ( (1-γ) \symbf v + γ \symbf u )ᵀ \symbf M ( (1-γ) \symbf v + γ \symbf u) \qquad  (γ ∈ [0,1])
```

We have
```math
σ(γ) \begin{aligned}[t]
	&=
	( (1-γ) \symbf v + γ \symbf u )ᵀ\symbf{M} ( (1-γ) \symbf v + γ \symbf{u})
		\\
	&=
	(1-γ)² \underbrace{\symbf{v}ᵀ\symbf{M} \symbf{v}}_{a} +
	  2γ(1-γ) \underbrace{\symbf{u}ᵀ\symbf{M} \symbf{v}}_{b} +
	    γ² \underbrace{\symbf{u}ᵀ\symbf{M} \symbf{u}}_{c}
		\\
	&=
	(1 + γ² - 2γ)a + (2γ - 2γ²)b + γ² c
		\\
	&=
	(a -2b + c) γ² + 2 (b-a) γ + a
\end{aligned}
```
The variables $a, b$ and $c$ are scalar.
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

````@example multidir_frank_wolfe
"""
	min_quad(a,b,c)

Given a quadratic function ``(a -2b + c) γ² + 2 (b-a) γ + a`` with ``γ ∈ [0,1]``, return
`γ_opt` minimizing the function in that interval and its optimal value `σ_opt`.
"""
function min_quad(a,b,c)
	a_min_b = a-b
	b_min_c = b-c
	if a_min_b > b_min_c
		# the function is a convex parabola and has its global minimum at `γ`
		γ = a_min_b /(a_min_b - b_min_c)
		if 0 < γ < 1
			# if its in the interval, return it
			σ = a - a_min_b * γ
			return γ, σ
		end
	end
	# the function is either a line or a concave parabola, the minimum is attained at the
	# boundaries
	if a <= c
		return 0, a
	else
		return 1, c
	end
end
````

To use the above function in the Frank-Wolfe algorithm, we define a
helper according to the definitions of ``a,b`` and ``c``:

````@example multidir_frank_wolfe
function min_chull2(M, v, u)
	Mv = M*v
	a = v'Mv
	b = u'Mv
	c = u'M*u
	return min_quad(a,b,c)
end
````

## Completed Algorithm

The stepsize computation is the most difficult part.
Now, we only have to care about stopping and can complete the solver
for our sub-problem:

````@example multidir_frank_wolfe
import LinearAlgebra as LA
````

````@example multidir_frank_wolfe
function frank_wolfe_multidir_dual(grads; max_iter=10_000, eps_abs=1e-5)

	num_objfs = length(grads)
	T = Base.promote_type(Float32, mapreduce(eltype, promote_type, grads))

	# 1) Initialize ``α`` vector. There are smarter ways to do this...
	α = fill(T(1/num_objfs), num_objfs)

	# 2) Build symmetric matrix of gradient-gradient products
	# # `_M` will be a temporary, upper triangular matrix
	_M = zeros(T, num_objfs, num_objfs)
	for (i,gi) = enumerate(grads)
		for (j, gj) = enumerate(grads)
			j<i && continue
			_M[i,j] = gi'gj
		end
	end
	# # mirror `_M` to get the full symmetric matrix
	M = LA.Symmetric(_M, :U)

	# 3) Solver iteration
	_α = copy(α)    		# to keep track of change
	u = zeros(T, num_objfs) # seed vector
	for _=1:max_iter
		t = argmin( M*α )
		v = α
		fill!(u, 0)
		u[t] = one(T)

		γ, _ = min_chull2(M, v, u)

		α .*= (1-γ)
		α[t] += γ

		if sum( abs.( _α .- α ) ) <= eps_abs
			break
		end
		_α .= α
	end

	# return -sum(α .* grads) # somehow, broadcasting leads to type instability here,
	# see also https://discourse.julialang.org/t/type-stability-issues-when-broadcasting/92715
	return mapreduce(*, +, α, grads)
end
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

