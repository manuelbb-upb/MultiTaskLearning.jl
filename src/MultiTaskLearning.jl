module MultiTaskLearning

import MLDatasets: MNIST

include("MultiMNISTUtils.jl")
import .MultiMNISTUtils: MultiMNIST
export MultiMNIST

import Lux
import Lux: Chain, Conv, MaxPool, Dense, FlattenLayer, Dropout
import Lux: relu, logsoftmax

import OneHotArrays: onehotbatch

struct LRModel{B,L,R} <: Lux.AbstractExplicitContainerLayer{(:base, :l, :r)}
    base :: B
    l :: L
    r :: R
end

# setup a model like in 
# https://github.com/isl-org/MultiObjectiveOptimization/blob/master/multi_task/models/multi_lenet.py
function LRModel(;imgsize=(28,28,1), nclasses=10)
    dense_in = (imgsize[1]/4 - 3, imgsize[2]/4 - 3, 20)
    mod_base = Chain(
        Conv((5,5), last(imgsize)=>10, relu),   # WHCN order, (i1,i2,1,N) => (i1-5+1,i2-5+1,10,N)
        MaxPool((2,2)),                         # (i1-4, i2-4, 10, N) => (i1/2-2, i2/2-2, 10, N)
        Conv((5,5), 10=>20, relu),              # (i1/2-2, i2/2-2, 10, N) => (i1/2-2-5+1, i2/2-2-5+1, N)
        # NOTE is this what is happening?
        # compare https://github.com/isl-org/MultiObjectiveOptimization/blob/d45eb262ec61c0dafecebfb69027ff6de280dbb3/multi_task/models/multi_lenet.py#L17
        Dropout(0.5f0, 1f0, 3), # drop entire channels with probability 1/2, but don't scale remaining data...
        MaxPool((2,2)),                         # (i1/2-6, i2/2-6, N) => (i1/4-3, i2/4-3, 20, N)
        FlattenLayer(),
        Dense(Int(prod(dense_in)) => 50, relu)
    )
    mod_task = Chain(
        Dense(50 => 50, relu),
        Dropout(0.5f0, 1f0, :),
        Dense(50 => nclasses),
        #(x, ps, st) -> (logsoftmax(x), st), # does not work
        # Lux.WrappedFunction(logsoftmax) # TODO remove and use logit-loss
    )
    return LRModel(mod_base, mod_task, mod_task)
end

function (model::LRModel)(x::AbstractArray, ps, st::NamedTuple)
    z, st_base = model.base(x, ps.base, st.base)
    y_l, st_l = model.l(z, ps.l, st.l)
    y_r, st_r = model.l(z, ps.r, st.r)
    return (y_l, y_r), (base = st_base, l = st_l, r = st_r)
end

export LRModel
end # module MultiTaskLearning
