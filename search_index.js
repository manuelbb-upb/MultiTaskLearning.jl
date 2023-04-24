var documenterSearchIndex = {"docs":
[{"location":"exps/multidir_descent/","page":"Two-Task Learning with a Special LeNet","title":"Two-Task Learning with a Special LeNet","text":"EditURL = \"https://github.com/manuelbb-upb/MultiTaskLearning.jl/blob/main/exps/multidir_descent.jl\"","category":"page"},{"location":"exps/multidir_descent/#Two-Task-Learning-with-a-Special-LeNet","page":"Two-Task Learning with a Special LeNet","title":"Two-Task Learning with a Special LeNet","text":"","category":"section"},{"location":"exps/multidir_descent/","page":"Two-Task Learning with a Special LeNet","title":"Two-Task Learning with a Special LeNet","text":"Activate the environment to make dependencies available:","category":"page"},{"location":"exps/multidir_descent/","page":"Two-Task Learning with a Special LeNet","title":"Two-Task Learning with a Special LeNet","text":"using Pkg\nPkg.activate(\"/home/runner/work/MultiTaskLearning.jl/MultiTaskLearning.jl/docs/../exps\");\nnothing #hide","category":"page"},{"location":"exps/multidir_descent/","page":"Two-Task Learning with a Special LeNet","title":"Two-Task Learning with a Special LeNet","text":"This sript demonstrates how to apply","category":"page"},{"location":"exps/multidir_descent/","page":"Two-Task Learning with a Special LeNet","title":"Two-Task Learning with a Special LeNet","text":"standard multi-objective steepest descent and\n“partial” steepest descent","category":"page"},{"location":"exps/multidir_descent/","page":"Two-Task Learning with a Special LeNet","title":"Two-Task Learning with a Special LeNet","text":"to a classification problem with two objectives. In fact, we want to reproduce the results from [1].","category":"page"},{"location":"exps/multidir_descent/#Data-Loading-and-Pre-Processing","page":"Two-Task Learning with a Special LeNet","title":"Data Loading and Pre-Processing","text":"","category":"section"},{"location":"exps/multidir_descent/","page":"Two-Task Learning with a Special LeNet","title":"Two-Task Learning with a Special LeNet","text":"For that, we load a special MNIST Dataset:","category":"page"},{"location":"exps/multidir_descent/","page":"Two-Task Learning with a Special LeNet","title":"Two-Task Learning with a Special LeNet","text":"import MultiTaskLearning: MultiMNIST","category":"page"},{"location":"exps/multidir_descent/","page":"Two-Task Learning with a Special LeNet","title":"Two-Task Learning with a Special LeNet","text":"For plotting, let's simply use UnicodePlots.","category":"page"},{"location":"exps/multidir_descent/","page":"Two-Task Learning with a Special LeNet","title":"Two-Task Learning with a Special LeNet","text":"import UnicodePlots: heatmap","category":"page"},{"location":"exps/multidir_descent/","page":"Two-Task Learning with a Special LeNet","title":"Two-Task Learning with a Special LeNet","text":"Before visually inspecting the dataset, we would like to have it in the right format already. We use DataLoader for iteration and one-hot encoding for the labels:","category":"page"},{"location":"exps/multidir_descent/","page":"Two-Task Learning with a Special LeNet","title":"Two-Task Learning with a Special LeNet","text":"import MLUtils: DataLoader\nimport OneHotArrays: onehotbatch, onecold","category":"page"},{"location":"exps/multidir_descent/","page":"Two-Task Learning with a Special LeNet","title":"Two-Task Learning with a Special LeNet","text":"The following helper function loads the data and puts everything in order:","category":"page"},{"location":"exps/multidir_descent/","page":"Two-Task Learning with a Special LeNet","title":"Two-Task Learning with a Special LeNet","text":"function load_multi_mnist(mode=:train; batchsize=-1, shuffle=true, data_percentage=0.1)\n    if mode != :train mode = :test end\n\n    mmnist = MultiMNIST(Float32, mode)\n\n    last_index = min(length(mmnist), floor(Int, length(mmnist)*data_percentage))\n    imgs, _labels = mmnist[1:last_index];# _labels[:,i] is a vector with two labels for sample i\n    labels = onehotbatch(_labels, 0:9)   # labels[:,:,i] is a matrix, labels[:, 1, i] is the one hot vector of the first label, labels[:,2,i] is for the second label\n    sx, sy, num_dat = size(imgs)\n\n    # reshape for convolutional layer, it wants an additional dimension:\n    X = reshape(imgs, sx, sy, 1, num_dat)\n    return DataLoader(\n        (features=X, labels=(llabels=labels[:,1,:], rlabels=labels[:,2,:]));\n        shuffle, batchsize=batchsize > 0 ? min(batchsize,num_dat) : num_dat\n    )\nend","category":"page"},{"location":"exps/multidir_descent/","page":"Two-Task Learning with a Special LeNet","title":"Two-Task Learning with a Special LeNet","text":"If dat is a DataLoader returned by load_multi_mnist, then we can access the features of a batch with the features.property. For a single sample, this is an matrix, but for a batch its a multi-dimensional array, where the last index iterates samples. That's why we also define the following plotting helpers. The first one is for a single sample:","category":"page"},{"location":"exps/multidir_descent/","page":"Two-Task Learning with a Special LeNet","title":"Two-Task Learning with a Special LeNet","text":"function plot_mnist(\n    mat::AbstractMatrix, labels::Union{AbstractVector, Nothing}=nothing\n)\n    additional_settings = Dict()\n    if !isnothing(labels)\n        additional_settings[:title] = \"$(labels)\"\n    end\n    return heatmap(\n        mat'; yflip=true, colorbar=false, colormap=:grays, additional_settings...\n    )\nend","category":"page"},{"location":"exps/multidir_descent/","page":"Two-Task Learning with a Special LeNet","title":"Two-Task Learning with a Special LeNet","text":"And the second plotting function acts on a batch with extended dimensions. i is the sample index within the batch.","category":"page"},{"location":"exps/multidir_descent/","page":"Two-Task Learning with a Special LeNet","title":"Two-Task Learning with a Special LeNet","text":"function plot_mnist(\n    arr::AbstractArray{<:Real, 4}, Y::Union{NamedTuple, Nothing} = nothing;\n    i=1\n)\n    @assert size(arr, 3) == 1\n    @assert isnothing(Y) || (haskey(Y, :llabels) && haskey(Y, :rlabels))\n    if !isnothing(Y)\n        Yl = onecold(Y.llabels[:,i], 0:9)\n        Yr = onecold(Y.rlabels[:,i], 0:9)\n        Y = [Yl, Yr]\n    end\n    plot_mnist(arr[:,:,1,i], Y)\nend\n\nfunction plot_mnist( #hide\n    arr::AbstractArray{<:Real, 3}, Y::Union{AbstractMatrix, Nothing}=nothing; i=1 #hide\n)#hide\n    if !isnothing(Y)#hide\n        Y = onecold(Y, 0:9)#hide\n    end#hide\n    plot_mnist(arr[:,:,i], Y)#hide\nend#hide","category":"page"},{"location":"exps/multidir_descent/","page":"Two-Task Learning with a Special LeNet","title":"Two-Task Learning with a Special LeNet","text":"Let's finally have a look into the dataset:","category":"page"},{"location":"exps/multidir_descent/","page":"Two-Task Learning with a Special LeNet","title":"Two-Task Learning with a Special LeNet","text":"dat = load_multi_mnist(;batchsize=64);\nX, Y = first(dat); # extract first batch\nplot_mnist(X, Y)","category":"page"},{"location":"exps/multidir_descent/#Setting-up-the-Neural-Network.","page":"Two-Task Learning with a Special LeNet","title":"Setting up the Neural Network.","text":"","category":"section"},{"location":"exps/multidir_descent/","page":"Two-Task Learning with a Special LeNet","title":"Two-Task Learning with a Special LeNet","text":"To work with this specific data we can use the custom “Left-Right-Model” from the MultiTaskLearning package. It is a two-output LeNet architecture with some shared parameters for both outputs.","category":"page"},{"location":"exps/multidir_descent/","page":"Two-Task Learning with a Special LeNet","title":"Two-Task Learning with a Special LeNet","text":"import MultiTaskLearning: LRModel","category":"page"},{"location":"exps/multidir_descent/","page":"Two-Task Learning with a Special LeNet","title":"Two-Task Learning with a Special LeNet","text":"To set it up, we also need the Random module and Lux. Lux can either be added as a dependency to the environment or imported from MultiTaskLearning.","category":"page"},{"location":"exps/multidir_descent/","page":"Two-Task Learning with a Special LeNet","title":"Two-Task Learning with a Special LeNet","text":"import Random\nimport MultiTaskLearning: Lux","category":"page"},{"location":"exps/multidir_descent/","page":"Two-Task Learning with a Special LeNet","title":"Two-Task Learning with a Special LeNet","text":"We can now initialize a model and its parameters and states:","category":"page"},{"location":"exps/multidir_descent/","page":"Two-Task Learning with a Special LeNet","title":"Two-Task Learning with a Special LeNet","text":"rng = Random.seed!(31415)   # reproducible pseudo-random numbers\n# Initialize ann with shared parameters\nnn = LRModel();\nps, st = Lux.setup(rng, nn);    # parameters and states\nnothing #hide","category":"page"},{"location":"exps/multidir_descent/#Loss-and-Gradients","page":"Two-Task Learning with a Special LeNet","title":"Loss & Gradients","text":"","category":"section"},{"location":"exps/multidir_descent/","page":"Two-Task Learning with a Special LeNet","title":"Two-Task Learning with a Special LeNet","text":"We offer the multidir function to compute the multi-objective steepest descent direction from a jacobian matrix:","category":"page"},{"location":"exps/multidir_descent/","page":"Two-Task Learning with a Special LeNet","title":"Two-Task Learning with a Special LeNet","text":"import MultiTaskLearning: multidir","category":"page"},{"location":"exps/multidir_descent/","page":"Two-Task Learning with a Special LeNet","title":"Two-Task Learning with a Special LeNet","text":"The multidir function should work well with ComponentArrays, which allow for indexing the derivatives by the parameter names of the model.","category":"page"},{"location":"exps/multidir_descent/","page":"Two-Task Learning with a Special LeNet","title":"Two-Task Learning with a Special LeNet","text":"import ComponentArrays: ComponentArray, getaxes, Axis","category":"page"},{"location":"exps/multidir_descent/","page":"Two-Task Learning with a Special LeNet","title":"Two-Task Learning with a Special LeNet","text":"Of course, we also need some way of computing the loss derivatives, and we can use Zygote for this:","category":"page"},{"location":"exps/multidir_descent/","page":"Two-Task Learning with a Special LeNet","title":"Two-Task Learning with a Special LeNet","text":"import Zygote: withgradient, withjacobian, pullback, jacobian\n# optional: skip certain parts of code in gradient computation:\nimport ChainRulesCore: @ignore_derivatives, ignore_derivatives","category":"page"},{"location":"exps/multidir_descent/#Loss-Function(s)","page":"Two-Task Learning with a Special LeNet","title":"Loss Function(s)","text":"","category":"section"},{"location":"exps/multidir_descent/","page":"Two-Task Learning with a Special LeNet","title":"Two-Task Learning with a Special LeNet","text":"As ist customary with classification, we use logit-crossentropy, handcrafted from optimized functions in Lux:","category":"page"},{"location":"exps/multidir_descent/","page":"Two-Task Learning with a Special LeNet","title":"Two-Task Learning with a Special LeNet","text":"logitcrossentropy(y_pred, y) = Lux.mean(-sum(y .* Lux.logsoftmax(y_pred); dims=1))","category":"page"},{"location":"exps/multidir_descent/","page":"Two-Task Learning with a Special LeNet","title":"Two-Task Learning with a Special LeNet","text":"For best performance (LRModel does not cache (yet)), we'd like to have the loss for the left and right classification in one go. compute_losses does this and also returns the new network states, to suit a typical Lux workflow:","category":"page"},{"location":"exps/multidir_descent/","page":"Two-Task Learning with a Special LeNet","title":"Two-Task Learning with a Special LeNet","text":"function compute_losses(nn, ps, st, X, Y)\n    Y_pred, _st = Lux.apply(nn, X, ps, st)\n    losses = [\n        logitcrossentropy(Y_pred[1], Y.llabels);\n        logitcrossentropy(Y_pred[2], Y.rlabels)\n    ]\n    return losses, _st\nend","category":"page"},{"location":"exps/multidir_descent/#Testing-Things","page":"Two-Task Learning with a Special LeNet","title":"Testing Things","text":"","category":"section"},{"location":"exps/multidir_descent/","page":"Two-Task Learning with a Special LeNet","title":"Two-Task Learning with a Special LeNet","text":"We can now already most things on the first batch. The only thing left to do is wrapping the parameters as a ComponentVector.","category":"page"},{"location":"exps/multidir_descent/","page":"Two-Task Learning with a Special LeNet","title":"Two-Task Learning with a Special LeNet","text":"ps_c = ComponentArray(ps);\nnothing #hide","category":"page"},{"location":"exps/multidir_descent/","page":"Two-Task Learning with a Special LeNet","title":"Two-Task Learning with a Special LeNet","text":"Now, the initial derivative computation will take some time. Take note, that the Zygote methods return tuples, so we additionally pipe to first to get a matrix:","category":"page"},{"location":"exps/multidir_descent/","page":"Two-Task Learning with a Special LeNet","title":"Two-Task Learning with a Special LeNet","text":"jac = jacobian(params -> first(compute_losses(nn, params, st, X, Y)), ps_c) |> first;\ndir = multidir(jac)\nnew_ps_c = ps_c .+ dir","category":"page"},{"location":"exps/multidir_descent/#Training-Functions","page":"Two-Task Learning with a Special LeNet","title":"Training Functions","text":"","category":"section"},{"location":"exps/multidir_descent/","page":"Two-Task Learning with a Special LeNet","title":"Two-Task Learning with a Special LeNet","text":"Of course, for training we put this into functions. mode is a value reference to distinguish different strategies on dispatch level.","category":"page"},{"location":"exps/multidir_descent/","page":"Two-Task Learning with a Special LeNet","title":"Two-Task Learning with a Special LeNet","text":"import LinearAlgebra as LA\nfunction losses_states_jac(nn, ps_c, st, X, Y, mode=Val(:standard)::Val{:standard}; norm_grads=false)\n    local new_st\n    losses, jac_t = withjacobian(ps_c) do params\n        losses, new_st = compute_losses(nn, params, st, X, Y)\n        losses\n    end\n    jac = only(jac_t)\n    if norm_grads\n        jac ./= LA.norm.(eachrow(jac))\n    end\n    return losses, new_st, jac\nend\n\nfunction apply_multidir(\n    nn, ps_c, st, X, Y, mode=Val(:full)::Val{:standard};\n    lr=Float32(1e-3), jacmode=:standard, norm_grads=false\n)\n    losses, new_st, jac = losses_states_jac(nn, ps_c, st, X, Y, Val(jacmode); norm_grads)\n    dir = multidir(jac)\n    return losses, ps_c .+ lr .* dir, new_st\nend","category":"page"},{"location":"exps/multidir_descent/","page":"Two-Task Learning with a Special LeNet","title":"Two-Task Learning with a Special LeNet","text":"In the main train function, mode becomes a keyword argument:","category":"page"},{"location":"exps/multidir_descent/","page":"Two-Task Learning with a Special LeNet","title":"Two-Task Learning with a Special LeNet","text":"import Printf: @sprintf\nfunction train(\n    nn, ps_c, st, dat;\n    norm_grads=false,\n    dirmode=:full, jacmode=:standard, num_epochs=1, lr=Float32(1e-3)\n)\n    # printing offsets:\n    epad = ndigits(num_epochs)\n    num_batches = length(dat)\n    bpad = ndigits(num_batches)\n    # safeguard learning type\n    lr = eltype(ps_c)(lr)\n    # training loop\n    for e_ind in 1:num_epochs\n        @info \"----------- Epoch $(lpad(e_ind, epad)) ------------\"\n        epoch_stats = @timed for (b_ind, (X, Y)) in enumerate(dat)\n            batch_stats = @timed begin\n                losses, ps_c, st = apply_multidir(nn, ps_c, st, X, Y, Val(dirmode); lr, jacmode, norm_grads)\n                # excuse this ugly info string, please...\n                @info \"\\tE/B/Prog $(lpad(e_ind, epad)) / $(lpad(b_ind, bpad)) / $(@sprintf \"%3.2f\" b_ind/num_batches) %; l1 $(@sprintf \"%3.4f\" losses[1]); l2 $(@sprintf \"%3.4f\" losses[2])\"\n            end\n            @info \"\\t\\tBatch time: $(@sprintf \"%8.2f msecs\" batch_stats.time*1000).\"\n            if b_ind >= 2\n                break\n            end\n        end\n        @info \"Epoch time: $(@sprintf \"%.2f secs\" epoch_stats.time)\"\n    end\n    return ps_c, st\nend\n\nps_fin, st_fin = train(nn, ps_c, st, dat);\nnothing #hide","category":"page"},{"location":"exps/multidir_descent/#Partial-Multi-Descent","page":"Two-Task Learning with a Special LeNet","title":"Partial Multi-Descent","text":"","category":"section"},{"location":"exps/multidir_descent/","page":"Two-Task Learning with a Special LeNet","title":"Two-Task Learning with a Special LeNet","text":"In [1], the authors don't compute the multiobjective steepest descent direction with respect to all parameters, but only for the shared ones. Updates for the other parameters are performed with task-specific gradients:","category":"page"},{"location":"exps/multidir_descent/","page":"Two-Task Learning with a Special LeNet","title":"Two-Task Learning with a Special LeNet","text":"function apply_multidir(\n    nn, ps_c, st, X, Y, mode::Val{:partial};\n    lr=Float32(1e-3), jacmode=:standard, norm_grads=false\n)\n    losses, new_st, jac = losses_states_jac(nn, ps_c, st, X, Y, Val(jacmode); norm_grads)\n    # turn `jac` into ComponentMatrix; this allows for **very** convenient indexing\n    # `jac_c[:l, :]` is the gradient for the first task, `jac_c[:r, :]` for the second task\n    # `jac_c[:l, :base]` is that part of the gradient corresponding to shared parameters\n    # `jac_c[:l, :l]` is that part of the gradient corresponding to task-specific parameters\n    ax = only(getaxes(ps_c))\n    jac_c = ComponentArray(jac, Axis(l=1, r=2), ax)\n\n    # for the task parameters, apply specific parts of gradients\n    new_ps_c = deepcopy(ps_c)\n    new_ps_c.l .-= lr .* jac_c[:l, :l]\n    new_ps_c.r .-= lr .* jac_c[:r, :r]\n\n    # compute multi-objective steepest descent direction w.r.t. **shared** parameters\n    d = multidir(jac_c[:, :base])   ## NOTE the sub-component-matrix behaves like a normal matrix and can be provided to the solver :)\n    # apply multi-dir to shared parameters in-place\n    new_ps_c.base .+= lr .* d\n    return losses, new_ps_c, new_st\nend","category":"page"},{"location":"exps/multidir_descent/","page":"Two-Task Learning with a Special LeNet","title":"Two-Task Learning with a Special LeNet","text":"To train with this direction, we can call train(nn, ps_c, st, dat; dirmode=:partial).","category":"page"},{"location":"exps/multidir_descent/#Structured-Gradients:","page":"Two-Task Learning with a Special LeNet","title":"Structured Gradients:","text":"","category":"section"},{"location":"exps/multidir_descent/","page":"Two-Task Learning with a Special LeNet","title":"Two-Task Learning with a Special LeNet","text":"We are wasting a bit of memory for zeros introduced by the model structure. Maybe, we can compute the gradients more effectively. This is a ToDo","category":"page"},{"location":"exps/multidir_descent/","page":"Two-Task Learning with a Special LeNet","title":"Two-Task Learning with a Special LeNet","text":"[1]: O. Sener and V. Koltun, “Multi-Task Learning as Multi-Objective Optimization,”","category":"page"},{"location":"exps/multidir_descent/","page":"Two-Task Learning with a Special LeNet","title":"Two-Task Learning with a Special LeNet","text":"arXiv:1810.04650 [cs, stat], Jan. 2019, Accessed: Jan. 24, 2022. [Online]. Available: http://arxiv.org/abs/1810.04650","category":"page"},{"location":"exps/multidir_descent/","page":"Two-Task Learning with a Special LeNet","title":"Two-Task Learning with a Special LeNet","text":"","category":"page"},{"location":"exps/multidir_descent/","page":"Two-Task Learning with a Special LeNet","title":"Two-Task Learning with a Special LeNet","text":"This page was generated using Literate.jl.","category":"page"},{"location":"multidir_frank_wolfe/","page":"Custom Frank-Wolfe Solver...","title":"Custom Frank-Wolfe Solver...","text":"EditURL = \"https://github.com/manuelbb-upb/MultiTaskLearning.jl/blob/main/src/multidir_frank_wolfe.jl\"","category":"page"},{"location":"multidir_frank_wolfe/#Custom-Frank-Wolfe-Solver...","page":"Custom Frank-Wolfe Solver...","title":"Custom Frank-Wolfe Solver...","text":"","category":"section"},{"location":"multidir_frank_wolfe/","page":"Custom Frank-Wolfe Solver...","title":"Custom Frank-Wolfe Solver...","text":"... to compute the multi-objective steepest descent direction cheaply. For unconstrained problems, the direction can be computed by projecting symbf0in ℝ^n onto the convex hull of the negative objective gradients. This can be done easily with JuMP and a suitable solver (e.g., COSMO).","category":"page"},{"location":"multidir_frank_wolfe/","page":"Custom Frank-Wolfe Solver...","title":"Custom Frank-Wolfe Solver...","text":"In “Multi-Task Learning as Multi-Objective Optimization” by Sener & Koltun, the authors employ a variant of the Frank-Wolfe-type algorithms defined in “Revisiting Frank-Wolfe: Projection-Free Sparse Convex Optimization” by Jaggi.","category":"page"},{"location":"multidir_frank_wolfe/","page":"Custom Frank-Wolfe Solver...","title":"Custom Frank-Wolfe Solver...","text":"The objective for the projection problem is","category":"page"},{"location":"multidir_frank_wolfe/","page":"Custom Frank-Wolfe Solver...","title":"Custom Frank-Wolfe Solver...","text":"F(symbfα)\n= frac12  sum_i=1^K αᵢ fᵢ _2^2\n= frac12  nabla symbff^T symbfα _2^2\n= frac12 symbf a^T nabla symbff nabla symbff^T symbf α","category":"page"},{"location":"multidir_frank_wolfe/","page":"Custom Frank-Wolfe Solver...","title":"Custom Frank-Wolfe Solver...","text":"Hence,","category":"page"},{"location":"multidir_frank_wolfe/","page":"Custom Frank-Wolfe Solver...","title":"Custom Frank-Wolfe Solver...","text":"nabla F(symbfα)\n= nabla symbff nabla symbff^T symbf α\n= symbf M symbf α","category":"page"},{"location":"multidir_frank_wolfe/","page":"Custom Frank-Wolfe Solver...","title":"Custom Frank-Wolfe Solver...","text":"The algorithm starts with some initial symbfα = α_1  α_K^T and optimizes F within the standard simplex S = symbf α = α_1  α_k α_i ge 0 sum_i α_i = 1 This leads to the following procedure:","category":"page"},{"location":"multidir_frank_wolfe/","page":"Custom Frank-Wolfe Solver...","title":"Custom Frank-Wolfe Solver...","text":"Compute seed s as the minimizer of langle symbf s nabla F(symbf α_k) rangle = langle symbf s symbf M symbf α_krangle in S. The minimum is attained in one of the corners, i.e., symbf s = symbf e_t, where t is the minimizing index for the entries of symbf M symbf α_k.\nCompute the exact stepsize γin01 that minimizes\nF((1-γ)symbf α_k + γ symbf s)\nSet symbf α_k+1 = (1-γ_k) α_k + γ_k symbf s.","category":"page"},{"location":"multidir_frank_wolfe/#Finding-the-Stepsize","page":"Custom Frank-Wolfe Solver...","title":"Finding the Stepsize","text":"","category":"section"},{"location":"multidir_frank_wolfe/","page":"Custom Frank-Wolfe Solver...","title":"Custom Frank-Wolfe Solver...","text":"Let's discuss step 2. Luckily, we can indeed (easily) compute the minimizing stepsize. Suppose symbf v  ℝⁿ and symbf u  ℝⁿ are vectors and symbf M  ℝ^nn is a symmetric square matrix. What is the minimum of the following function?","category":"page"},{"location":"multidir_frank_wolfe/","page":"Custom Frank-Wolfe Solver...","title":"Custom Frank-Wolfe Solver...","text":"σ(γ) = ( (1-γ) symbf v + γ symbf u )ᵀ symbf M ( (1-γ) symbf v + γ symbf u) qquad  (γ  01)","category":"page"},{"location":"multidir_frank_wolfe/","page":"Custom Frank-Wolfe Solver...","title":"Custom Frank-Wolfe Solver...","text":"We have","category":"page"},{"location":"multidir_frank_wolfe/","page":"Custom Frank-Wolfe Solver...","title":"Custom Frank-Wolfe Solver...","text":"σ(γ) beginalignedt\n\t=\n\t( (1-γ) symbf v + γ symbf u )ᵀsymbfM ( (1-γ) symbf v + γ symbfu)\n\t\t\n\t=\n\t(1-γ)² underbracesymbfvᵀsymbfM symbfv_a +\n\t  2γ(1-γ) underbracesymbfuᵀsymbfM symbfv_b +\n\t    γ² underbracesymbfuᵀsymbfM symbfu_c\n\t\t\n\t=\n\t(1 + γ² - 2γ)a + (2γ - 2γ²)b + γ² c\n\t\t\n\t=\n\t(a -2b + c) γ² + 2 (b-a) γ + a\nendaligned","category":"page"},{"location":"multidir_frank_wolfe/","page":"Custom Frank-Wolfe Solver...","title":"Custom Frank-Wolfe Solver...","text":"The variables a b and c are scalar. The boundary values are","category":"page"},{"location":"multidir_frank_wolfe/","page":"Custom Frank-Wolfe Solver...","title":"Custom Frank-Wolfe Solver...","text":"σ₀ = σ(0) = a textand σ₁ = σ(1) = c","category":"page"},{"location":"multidir_frank_wolfe/","page":"Custom Frank-Wolfe Solver...","title":"Custom Frank-Wolfe Solver...","text":"If (a-2b+c)  0  a-b  b-c, then the parabola is convex and has its global minimum where the derivative is zero:","category":"page"},{"location":"multidir_frank_wolfe/","page":"Custom Frank-Wolfe Solver...","title":"Custom Frank-Wolfe Solver...","text":"2(a - 2b + c) y^* + 2(b-a) stackrel= 0\n \n\tγ^* = frac-2(b-a)2(a -2 b + c)\n\t\t= fraca-b(a-b)+(c-b)","category":"page"},{"location":"multidir_frank_wolfe/","page":"Custom Frank-Wolfe Solver...","title":"Custom Frank-Wolfe Solver...","text":"If a-b  b -c, the parabola is concave and this is a maximum. The extremal value is","category":"page"},{"location":"multidir_frank_wolfe/","page":"Custom Frank-Wolfe Solver...","title":"Custom Frank-Wolfe Solver...","text":"σ_* = σ(γ^*)\n\t= frac(a - b)^2(a-b)+(c-b) - frac2(a-b)^2(a-b) + (c-b) + a\n\t= a - frac(a-b)^2(a-b) + (c-b)","category":"page"},{"location":"multidir_frank_wolfe/","page":"Custom Frank-Wolfe Solver...","title":"Custom Frank-Wolfe Solver...","text":"\"\"\"\n\tmin_quad(a,b,c)\n\nGiven a quadratic function ``(a -2b + c) γ² + 2 (b-a) γ + a`` with ``γ ∈ [0,1]``, return\n`γ_opt` minimizing the function in that interval and its optimal value `σ_opt`.\n\"\"\"\nfunction min_quad(a,b,c)\n\ta_min_b = a-b\n\tb_min_c = b-c\n\tif a_min_b > b_min_c\n\t\t# the function is a convex parabola and has its global minimum at `γ`\n\t\tγ = a_min_b /(a_min_b - b_min_c)\n\t\tif 0 < γ < 1\n\t\t\t# if its in the interval, return it\n\t\t\tσ = a - a_min_b * γ\n\t\t\treturn γ, σ\n\t\tend\n\tend\n\t# the function is either a line or a concave parabola, the minimum is attained at the\n\t# boundaries\n\tif a <= c\n\t\treturn 0, a\n\telse\n\t\treturn 1, c\n\tend\nend","category":"page"},{"location":"multidir_frank_wolfe/","page":"Custom Frank-Wolfe Solver...","title":"Custom Frank-Wolfe Solver...","text":"To use the above function in the Frank-Wolfe algorithm, we define a helper according to the definitions of ab and c:","category":"page"},{"location":"multidir_frank_wolfe/","page":"Custom Frank-Wolfe Solver...","title":"Custom Frank-Wolfe Solver...","text":"function min_chull2(M, v, u)\n\tMv = M*v\n\ta = v'Mv\n\tb = u'Mv\n\tc = u'M*u\n\treturn min_quad(a,b,c)\nend","category":"page"},{"location":"multidir_frank_wolfe/#Completed-Algorithm","page":"Custom Frank-Wolfe Solver...","title":"Completed Algorithm","text":"","category":"section"},{"location":"multidir_frank_wolfe/","page":"Custom Frank-Wolfe Solver...","title":"Custom Frank-Wolfe Solver...","text":"The stepsize computation is the most difficult part. Now, we only have to care about stopping and can complete the solver for our sub-problem:","category":"page"},{"location":"multidir_frank_wolfe/","page":"Custom Frank-Wolfe Solver...","title":"Custom Frank-Wolfe Solver...","text":"import LinearAlgebra as LA","category":"page"},{"location":"multidir_frank_wolfe/","page":"Custom Frank-Wolfe Solver...","title":"Custom Frank-Wolfe Solver...","text":"function frank_wolfe_multidir_dual(grads; max_iter=10_000, eps_abs=1e-5)\n\n\tnum_objfs = length(grads)\n\tT = Base.promote_type(Float32, mapreduce(eltype, promote_type, grads))\n\n\t# 1) Initialize ``α`` vector. There are smarter ways to do this...\n\tα = fill(T(1/num_objfs), num_objfs)\n\n\t# 2) Build symmetric matrix of gradient-gradient products\n\t# # `_M` will be a temporary, upper triangular matrix\n\t_M = zeros(T, num_objfs, num_objfs)\n\tfor (i,gi) = enumerate(grads)\n\t\tfor (j, gj) = enumerate(grads)\n\t\t\tj<i && continue\n\t\t\t_M[i,j] = gi'gj\n\t\tend\n\tend\n\t# # mirror `_M` to get the full symmetric matrix\n\tM = LA.Symmetric(_M, :U)\n\n\t# 3) Solver iteration\n\t_α = copy(α)    \t\t# to keep track of change\n\tu = zeros(T, num_objfs) # seed vector\n\tfor _=1:max_iter\n\t\tt = argmin( M*α )\n\t\tv = α\n\t\tfill!(u, 0)\n\t\tu[t] = one(T)\n\n\t\tγ, _ = min_chull2(M, v, u)\n\n\t\tα .*= (1-γ)\n\t\tα[t] += γ\n\n\t\tif sum( abs.( _α .- α ) ) <= eps_abs\n\t\t\tbreak\n\t\tend\n\t\t_α .= α\n\tend\n\n\t# return -sum(α .* grads) # somehow, broadcasting leads to type instability here,\n\t# see also https://discourse.julialang.org/t/type-stability-issues-when-broadcasting/92715\n\treturn mapreduce(*, +, α, grads)\nend","category":"page"},{"location":"multidir_frank_wolfe/","page":"Custom Frank-Wolfe Solver...","title":"Custom Frank-Wolfe Solver...","text":"","category":"page"},{"location":"multidir_frank_wolfe/","page":"Custom Frank-Wolfe Solver...","title":"Custom Frank-Wolfe Solver...","text":"This page was generated using Literate.jl.","category":"page"},{"location":"#MultiTaskLearning.jl","page":"MultiTaskLearning.jl","title":"MultiTaskLearning.jl","text":"","category":"section"},{"location":"","page":"MultiTaskLearning.jl","title":"MultiTaskLearning.jl","text":"Documentation for MultiTaskLearning.jl","category":"page"}]
}
