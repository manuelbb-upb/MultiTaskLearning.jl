# This file is formatted to be parsed by "Literate.jl" #src

# Activate the environment to make dependencies available #src
using Pkg               #hide
Pkg.activate(@__DIR__)  #hide

# # Two-Task Learning with a Spcial LeNet

# This sript demonstrates how to apply 
# * standard multi-objective steepest descent and 
# * “partial” steepest descent 
# to a classification problem with two objectives.
# In fact, we want to reproduce the results from [^1].

# ## Data Loading and Pre-Processing
# For that, we load a special MNIST Dataset:
import MultiTaskLearning: MultiMNIST

# For plotting, let's simply use `UnicodePlots`.
import UnicodePlots: heatmap

# Before visually inspecting the dataset, we would like to have it 
# in the right format already.
# We use `DataLoader` for iteration and one-hot encoding for the labels:
import MLUtils: DataLoader
import OneHotArrays: onehotbatch, onecold

# The following helper function loads the data and puts everything in order:
function load_multi_mnist(mode=:train; batchsize=-1, shuffle=true, data_percentage=0.1)
    if mode != :train mode = :test end
    
    mmnist = MultiMNIST(Float32, mode)

    last_index = min(length(mmnist), floor(Int, length(mmnist)*data_percentage))
    imgs, _labels = mmnist[1:last_index];# _labels[:,i] is a vector with two labels for sample i
    labels = onehotbatch(_labels, 0:9)   # labels[:,:,i] is a matrix, labels[:, 1, i] is the one hot vector of the first label, labels[:,2,i] is for the second label
    sx, sy, num_dat = size(imgs)

    ## reshape for convolutional layer, it wants an additional dimension:
    X = reshape(imgs, sx, sy, 1, num_dat)
    return DataLoader(
        (features=X, labels=(llabels=labels[:,1,:], rlabels=labels[:,2,:])); 
        shuffle, batchsize=batchsize > 0 ? min(batchsize,num_dat) : num_dat
    )
end

# If `dat` is a DataLoader returned by `load_multi_mnist`, then we can access
# the features of a batch with the `features.property`.
# For a single sample, this is an matrix, but for a batch its a multi-dimensional
# array, where the last index iterates samples.
# That's why we also define the following plotting helpers.
# The first one is for a single sample:
function plot_mnist(
    mat::AbstractMatrix, labels::Union{AbstractVector, Nothing}=nothing
) 
    additional_settings = Dict()
    if !isnothing(labels)
        additional_settings[:title] = "$(labels)"
    end
    return heatmap(
        mat'; yflip=true, colorbar=false, colormap=:grays, additional_settings...
    )
end
# And the second plotting function acts on a batch with extended dimensions.
# `i` is the sample index within the batch.
function plot_mnist(
    arr::AbstractArray{<:Real, 4}, Y::Union{NamedTuple, Nothing} = nothing; 
    i=1
)
    @assert size(arr, 3) == 1
    @assert isnothing(Y) || (haskey(Y, :llabels) && haskey(Y, :rlabels))
    if !isnothing(Y)
        Yl = onecold(Y.llabels[:,i], 0:9)
        Yr = onecold(Y.rlabels[:,i], 0:9)
        Y = [Yl, Yr]
    end
    plot_mnist(arr[:,:,1,i], Y)
end

# (This function is kept for historic reasons, but not needed with DataLoader) #hide
function plot_mnist( #hide
    arr::AbstractArray{<:Real, 3}, Y::Union{AbstractMatrix, Nothing}=nothing; i=1 #hide
)#hide
    if !isnothing(Y)#hide
        Y = onecold(Y, 0:9)#hide
    end#hide
    plot_mnist(arr[:,:,i], Y)#hide
end#hide

# Let's finally have a look into the dataset:
dat = load_multi_mnist(;batchsize=64);
X, Y = first(dat); # extract first batch
plot_mnist(X, Y)

# ## Setting up the Neural Network.
# To work with this specific data we can use the custom “Left-Right-Model” 
# from the `MultiTaskLearning` package.
# It is a two-output LeNet architecture with some shared parameters for both outputs.
import MultiTaskLearning: LRModel
# To set it up, we also need the `Random` module and `Lux`.
# `Lux` can either be added as a dependency to the environment or imported 
# from `MultiTaskLearning`.
import Random
import MultiTaskLearning: Lux

# We can now initialize a model and its parameters and states:
rng = Random.seed!(31415)   # reproducible pseudo-random numbers
## Initialize ann with shared parameters
nn = LRModel();
ps, st = Lux.setup(rng, nn);    # parameters and states

# ## Loss & Gradients
# We offer the `multidir` function to compute the multi-objective steepest descent 
# direction from a jacobian matrix:
import MultiTaskLearning: multidir
# The `multidir` function should work well with `ComponentArray`s, which allow for 
# indexing the derivatives by the parameter names of the model.
import ComponentArrays: ComponentArray, getaxes, Axis
# Of course, we also need some way of computing the loss derivatives, and we can use 
# Zygote for this:
import Zygote: withgradient, withjacobian, pullback, jacobian
## optional: skip certain parts of code in gradient computation:
import ChainRulesCore: @ignore_derivatives, ignore_derivatives

# ### Loss Function(s)
# As ist customary with classification, we use logit-crossentropy, handcrafted 
# from optimized functions in `Lux`:
logitcrossentropy(y_pred, y) = Lux.mean(-sum(y .* Lux.logsoftmax(y_pred); dims=1))
# For best performance (`LRModel` does not cache (yet)), we'd like to have 
# the loss for the left and right classification in one go.
# `compute_losses` does this and also returns the new network states, to suit
# a typical `Lux` workflow:
function compute_losses(nn, ps, st, X, Y)
    Y_pred, _st = Lux.apply(nn, X, ps, st)
    losses = [ 
        logitcrossentropy(Y_pred[1], Y.llabels);
        logitcrossentropy(Y_pred[2], Y.rlabels)
    ]
    return losses, _st
end

# ### Testing Things
# We can now already most things on the first batch.
# The only thing left to do is wrapping the parameters as a `ComponentVector`.
ps_c = ComponentArray(ps);
# Now, the initial derivative computation will take some time.
# Take note, that the Zygote methods return tuples, so we additionally pipe to `first`
# to get a matrix:
jac = jacobian(params -> first(compute_losses(nn, params, st, X, Y)), ps_c) |> first;
dir = multidir(jac)
new_ps_c = ps_c .+ dir

# ### Training Functions
# Of course, for training we put this into functions.
# `mode` is a value reference to distinguish different strategies on dispatch level.
import LinearAlgebra as LA
function losses_states_jac(nn, ps_c, st, X, Y, mode=Val(:standard)::Val{:standard}; norm_grads=false)
    local new_st
    losses, jac_t = withjacobian(ps_c) do params
        losses, new_st = compute_losses(nn, params, st, X, Y)
        losses
    end
    jac = only(jac_t)
    if norm_grads
        jac ./= LA.norm.(eachrow(jac))
    end
    return losses, new_st, jac
end

function apply_multidir(
    nn, ps_c, st, X, Y, mode=Val(:full)::Val{:standard}; 
    lr=Float32(1e-3), jacmode=:standard, norm_grads=false
)
    losses, new_st, jac = losses_states_jac(nn, ps_c, st, X, Y, Val(jacmode); norm_grads)
    dir = multidir(jac)
    return losses, ps_c .+ lr .* dir, new_st
end

# In the main `train` function, mode becomes a keyword argument:
import Printf: @sprintf
function train(
    nn, ps_c, st, dat;
    norm_grads=false,
    dirmode=:full, jacmode=:standard, num_epochs=1, lr=Float32(1e-3)
)
    ## printing offsets:
    epad = ndigits(num_epochs)
    num_batches = length(dat)
    bpad = ndigits(num_batches)
    ## safeguard learning type
    lr = eltype(ps_c)(lr)
    ## training loop
    for e_ind in 1:num_epochs
        @info "----------- Epoch $(lpad(e_ind, epad)) ------------"
        epoch_stats = @timed for (b_ind, (X, Y)) in enumerate(dat)
            batch_stats = @timed begin
                losses, ps_c, st = apply_multidir(nn, ps_c, st, X, Y, Val(dirmode); lr, jacmode, norm_grads)
                ## excuse this ugly info string, please...
                @info "\tE/B/Prog $(lpad(e_ind, epad)) / $(lpad(b_ind, bpad)) / $(@sprintf "%3.2f" b_ind/num_batches) %; l1 $(@sprintf "%3.4f" losses[1]); l2 $(@sprintf "%3.4f" losses[2])"
            end
            @info "\t\tBatch time: $(@sprintf "%8.2f msecs" batch_stats.time*1000)."
            if b_ind >= 2
                break
            end
        end
        @info "Epoch time: $(@sprintf "%.2f secs" epoch_stats.time)"
    end
    return ps_c, st
end

ps_fin, st_fin = train(nn, ps_c, st, dat);

# ## Partial Multi-Descent
# In [^1], the authors don't compute the multiobjective steepest descent direction 
# with respect to all parameters, but only for the shared ones.
# Updates for the other parameters are performed with task-specific gradients:

function apply_multidir(
    nn, ps_c, st, X, Y, mode::Val{:partial}; 
    lr=Float32(1e-3), jacmode=:standard, norm_grads=false
)
    losses, new_st, jac = losses_states_jac(nn, ps_c, st, X, Y, Val(jacmode); norm_grads)
    ## turn `jac` into ComponentMatrix; this allows for **very** convenient indexing
    ## `jac_c[:l, :]` is the gradient for the first task, `jac_c[:r, :]` for the second task
    ## `jac_c[:l, :base]` is that part of the gradient corresponding to shared parameters
    ## `jac_c[:l, :l]` is that part of the gradient corresponding to task-specific parameters
    ax = only(getaxes(ps_c))
    jac_c = ComponentArray(jac, Axis(l=1, r=2), ax)

    ## for the task parameters, apply specific parts of gradients
    new_ps_c = deepcopy(ps_c)
    new_ps_c.l .-= lr .* jac_c[:l, :l]
    new_ps_c.r .-= lr .* jac_c[:r, :r]

    ## compute multi-objective steepest descent direction w.r.t. **shared** parameters
    d = multidir(jac_c[:, :base])   ## NOTE the sub-component-matrix behaves like a normal matrix and can be provided to the solver :)
    ## apply multi-dir to shared parameters in-place
    new_ps_c.base .+= lr .* d
    return losses, new_ps_c, new_st
end
# To train with this direction, we can call `train(nn, ps_c, st, dat; dirmode=:partial)`.

# ## Structured Gradients:
# We are wasting a bit of memory for zeros introduced by the model structure.
# Maybe, we can compute the gradients more effectively.
# **This is a ToDo**

# [^1]: O. Sener and V. Koltun, “Multi-Task Learning as Multi-Objective Optimization,” 
# arXiv:1810.04650 [cs, stat], Jan. 2019, Accessed: Jan. 24, 2022. [Online]. 
# Available: http://arxiv.org/abs/1810.04650