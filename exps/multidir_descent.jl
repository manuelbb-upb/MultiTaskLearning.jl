using Pkg
Pkg.activate(@__DIR__)

# Custom Multi-Task models
import MultiTaskLearning: LRModel, multidir
import MultiTaskLearning: Lux # for setting up models and defining loss # TODO consider re-export from within MultiTaskLearning

# Dataset loading & processing
import MLUtils: DataLoader
import OneHotArrays: onehotbatch, onecold
import MultiTaskLearning: MNIST, MultiMNIST

# Training and parameter manipulation
import Zygote: withgradient, withjacobian
import ComponentArrays: ComponentArray, getaxes, Axis

# stdlib libraries
import Random       # reproducible pseudo-random numbers
import LinearAlgebra as LA

# visualization
import UnicodePlots: heatmap
#%%
# Plotting helpers
function plot_mnist(mat::AbstractMatrix, labels::Union{AbstractVector, Nothing}=nothing)
    additional_settings = Dict()
    if !isnothing(labels)
        additional_settings[:title] = "$(labels)"
    end
    return heatmap(mat'; yflip=true, colorbar=false, colormap=:grays, additional_settings...)
end

function plot_mnist(arr::AbstractArray{<:Real, 3}, Y::Union{AbstractMatrix, Nothing}=nothing; i=1)
    if !isnothing(Y)
        Y = onecold(Y, 0:9)
    end
    plot_mnist(arr[:,:,i], Y)
end

function plot_mnist(arr::AbstractArray{<:Real, 4}, Y::Union{AbstractArray{<:Number, 3}, Nothing} = nothing; i=1)
    @assert size(arr, 3) == 1
    @assert isnothing(Y) || size(Y, 2) == 2
    if !isnothing(Y)
        Y = onecold(Y[:,:,i], 0:9)
    end
    plot_mnist(arr[:,:,1,i], Y)
end
#%%

function load_multi_mnist(mode=:train; batchsize=-1, shuffle=true)
    if mode != :train mode = :test end
    smnist = MNIST(mode)
    mmnist = MultiMNIST(smnist)#; llb=:reset)

    imgs, _labels = mmnist[:];          # _labels[:,i] is a vector with two labels for sample i
    labels = onehotbatch(_labels, 0:9)   # labels[:,:,i] is a matrix, labels[:, 1, i] is the one hot vector of the first label, labels[:,2,i] is for the second label
    sx, sy, num_dat = size(imgs)

    # reshape for convolutional layer, which does not like matrices...
    X = reshape(imgs, sx, sy, 1, num_dat)
    return DataLoader(
        #(features=X, label1=labels[:,1,:], label2=labels[:,2,:]); 
        (features=X, labels=labels); 
        shuffle,
        batchsize=batchsize > 0 ? min(batchsize,num_dat) : num_dat
    )
end

#%%
# Training functions

# “Classical” (stochastic) steepest descent
function apply_multidir!(ps_c, st_ref, nn, X, Y; lr=1e-3)
    # Compute loss vector and jacobian w.r.t. parameters `ps_c` in one sweep:
    losses, jac_tuple = withjacobian(ps_c) do params
        l, _st = compute_losses(nn, params, st_ref[], X, Y)
        st_ref[] = _st
        return l
    end
    # extract jacobian matrix
    jac = only(jac_tuple)
    # compute multi-objective steepest descent direction w.r.t. all parameters
    d = multidir(jac)

    # apply direction in place
    ps_c .+= lr .* d

    return losses
end

function apply_partial_multidir!(ps_c, st_ref, nn, X, Y, ax=nothing; lr=1e-3)
    __ax = isnothing(ax) ? getaxes(ps_c) : ax
    _ax = __ax isa Tuple ? first(__ax) : __ax

    # Compute loss vector and jacobian w.r.t. parameters `ps_c` in one sweep:
    losses, jac_tuple = withjacobian(ps_c) do params
        l, _st = compute_losses(nn, params, st_ref[], X, Y)
        st_ref[] = _st
        return l
    end
    # extract jacobian matrix
    jac = only(jac_tuple)
    # turn into ComponentMatrix; this allows for **very** convenient indexing
    ## `jac_c[:l, :]` is the gradient for the first task, `jac_c[:r, :]` for the second task
    ## `jac_c[:l, :base]` is that part of the gradient corresponding to shared parameters
    ## `jac_c[:l, :l]` is that part of the gradient corresponding to task-specific parameters
    jac_c = ComponentArray(jac, Axis(l=1, r=2), _ax)

    # for the task parameters, apply specific parts of gradients
    ps_c.l .-= lr .* jac_c[:l, :l]
    ps_c.r .-= lr .* jac_c[:r, :r]

    # compute multi-objective steepest descent direction w.r.t. **shared** parameters
    d = multidir(jac_c[:, :base])   # NOTE the sub-component-matrix behaves like a normal matrix and can be provided to the solver :)

    # apply multi-dir to shared parameters in-place
    ps_c.base .+= lr .* d
    
    return losses
end

function train_full!(ps_c, st_ref, nn, dat; lr=1e-3)
    for (X,Y) in dat
        losses = apply_multidir!(ps_c, st_ref, nn, X, Y; lr)
        @show losses
    end
end
function train_partial!(ps_c, st_ref, nn, dat; lr=1e-3)
    for (X,Y) in dat
        losses = apply_partial_multidir!(ps_c, st_ref, nn, X, Y; lr)
        @show losses
    end
end

#%%
logitcrossentropy(y_pred, y) = Lux.mean(-sum(y .* Lux.logsoftmax(y_pred); dims=1))

function compute_losses(nn, ps, st, X, Y)
    Y_pred, _st = Lux.apply(nn, X, ps, st)
    losses = [ logitcrossentropy(Y_pred[i], Y[:,i,:]) for i=eachindex(Y_pred) ]
    return losses, _st
end
#%%

rng = Random.seed!(31415)       # reproducible pseudo-random numbers

# initialize ann with shared parameters
nn = LRModel();
ps, st = Lux.setup(rng, nn);    # parameters and states
# wrap states in reference and turn parameters into Component Vector
st_ref = Ref(st);
ps_c = ComponentArray(ps);       # enables turning vectors into NamedTuple-like structures and vice-versa
ax = only(getaxes(ps_c));

dat = load_multi_mnist(;batchsize=100);

train_full!(ps_c, st_ref, nn, dat)
