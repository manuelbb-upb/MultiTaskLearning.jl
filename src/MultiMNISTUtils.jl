module MultiMNISTUtils

export MultiMNIST

import MLDatasets: MNIST, SupervisedDataset
import Interpolations: Constant
import GZip
import ImageTransformations: imresize
import Random: Xoshiro

# For a description of the "IDX" file format, see 
# http://yann.lecun.com/exdb/mnist/

# mapping of byte value to data type
const DATTYPE_DICT = Base.ImmutableDict(
    0x08 => UInt8,
    0x09 => Int8,
    0x0B => Int16,
    0x0C => Int32,
    0x0D => Float32,
    0x0E => Float64
)

# helper to read array of 4 bytes (UInt8) to an Int32
function read_int(bytes)
    @assert length(bytes) == 4
    return ntoh(only(reinterpret(Int32, bytes)))
end

# take bytes from IDX data and return a NamedTuple with some metadata and the parsed data
# content
function parse_idx(bytes)
    dtype = DATTYPE_DICT[bytes[3]]
    dims = Int(bytes[4])
    
    magic_number = read_int(bytes[1:4])
    offset = 4

    shape = zeros(Int32, dims)
    for i=1:dims
        shape[i] = read_int(bytes[offset+1:offset+4])
        offset += 4
    end

    # convert remaining bytes to array with correct data type and shape
    ## last dimension changes fastest: reverse shape
    shaped_data = reshape(
        ntoh.(reinterpret(dtype, bytes[offset+1:end])), 
        reverse(shape)...
    )

    return (
        magic_number = magic_number,
        dims = dims,
        shape = shape,
        dtype = dtype,
    ), shaped_data
end

function to_idx_bytes(arr)
    return reshape(reinterpret(UInt8, hton.(arr)), :)
end

function global_coordinates(
    lb;
    local_img_size=(28, 28), new_img_size=(36, 36)
)
    #=
        (LB[1], UB[2]) ----- (UB[1], UB[2])
                |                   |
                |                   |
        (LB[1], LB[2]) ----- (UB[1], LB[2])
    =#
    LB = (1,1)
    UB = new_img_size
    ub = lb .+ local_img_size .- 1
    _lb = min.(UB, max.(LB, lb))
    _ub = min.(UB, max.(LB,ub))
    _w = _ub .- _lb .+ 1
    return _lb, _ub, _w
end

function local_coordinates(lub, rlb, s1, s2, lw, rw)
    # local coordinates:
    __llb = _llb = (s1, s2) .- lw .+ 1
    __rlb = _rlb = (s1, s2) .- rw .+ 1
    __lub = __rub = _lub = _rub = (s1, s2)
    ## for overlap:
    d = lub .- rlb
    has_overlap = false
    if all(d .> 0)
        has_overlap = true
        __llb = _lub .- d
        __rub = _rlb .+ d
    end
    return has_overlap, _llb, _rlb, _lub, _rub, __llb, __rlb, __lub, __rub
end

inv_trans(_) = identity
inv_trans(::typeof(rotr90)) = rotl90
inv_trans(::typeof(rotl90)) = rotr90

predict_transformed_size(_, s1, s2) = (s1, s2) 
predict_transformed_size(::typeof(rotr90), s1, s2) = (s2, s1) 
predict_transformed_size(::typeof(rotl90), s1, s2) = (s2, s1) 

# 1) create a working area of size `tmp_size`
# 2) at projected lower bounds point `llb`:
#    a) apply `drawing_transformation` to left image
#    b) place its lower left corner at `llb`
#    Do the Same for `rlb`.
# 3) When the new image is finished:
#    a) resize it to `newimg_size`
#    b) apply reverse transformation
function make_new_imgs(
    fshape, fdata, tdata; 
    llb::Union{Symbol, Nothing, NTuple{2,<:Integer}}=nothing,
    rlb::Union{Symbol, Nothing, NTuple{2,<:Integer}}=nothing,
    newimg_size::Union{Nothing, NTuple{2,<:Integer}}=nothing,
    tmp_size::Union{Nothing, NTuple{2,<:Integer}}=nothing,
    drawing_transformation::Union{typeof(identity), typeof(rotr90), typeof(rotl90)}=rotl90
)
    rng = Xoshiro(31415)      # reproducibly choose RHS image
    
    #fdata = fdata[1:15,:,:]
    #fshape = reverse(size(fdata))

    dims = length(fshape)
    @assert dims == 3
    @assert all(size(fdata) .== reverse(fshape))

    len, _s2, _s1 = Int.(fshape)
    rot = drawing_transformation
    unrot = inv_trans(rot)

    # `(s1,s2)` is the size of a rotated original image
    s1, s2 = predict_transformed_size(rot, _s1, _s2)

    # llb = (1,1), rlb=(7,7), tmp_size=(36,36) is used by Sener & Koltun
    if isnothing(llb) || llb isa Symbol
        llb = (1,1)
    end
    if isnothing(rlb) || rlb isa Symbol
        rlb = (7,7)
    end
    if isnothing(newimg_size)
        newimg_size = (s1, s2)
    end
    if isnothing(tmp_size) || tmp_size isa Symbol
        tmp_size = newimg_size .+ 8
    end
    @assert all(rlb .>= llb)

    # `empty_new` is a template for the drawing area
    empty_new = zeros(eltype(fdata), tmp_size) # empty_new = unrot(zeros(eltype(fdata), tmp_size))
   	S1, S2 = size(empty_new)
    llb, lub, lw = global_coordinates(llb; local_img_size=(s1, s2), new_img_size=(S1, S2))
    rlb, rub, rw = global_coordinates(rlb; local_img_size=(s1, s2), new_img_size=(S1, S2))

    has_overlap, _llb, _rlb, _lub, _rub, __llb, __rlb, __lub, __rub = local_coordinates(lub, rlb, s1, s2, lw, rw)

    new_imgs = similar(fdata, predict_transformed_size(unrot, newimg_size...)..., len)
    new_targets = similar(tdata, (2, len))

    for ileft = 1:len
		iright = rand(rng, 1:len)    # choose random index for right feature img
		img_left = rot(fdata[:,:,ileft])
		img_right = rot(fdata[:,:,iright])
		img_new = copy(empty_new)
	    
        # the pixel `img_new[llb[1], llb[2]]` will have value `img_left[_llb[1], _llb[1]]`
        # likewise, `img_new[llb[1]:lub[1], llb[2]:lub[2]]` have values `img_left[_llb[1]:_lub[1], _llb[1]:_lub[2]]`
        # same for RHS
        img_new[llb[1]:lub[1], llb[2]:lub[2]] .= img_left[_llb[1]:_lub[1], _llb[2]:_lub[2]]
		img_new[rlb[1]:rub[1], rlb[2]:rub[2]] .= img_right[_rlb[1]:_rub[1], _rlb[2]:_rub[2]]
        
        # overlap: maximum value
        if has_overlap
		    img_new[rlb[1]:lub[1], rlb[2]:lub[2]] .= max.(
                img_left[__llb[1]:__lub[1], __llb[2]:__lub[2]],
                img_right[__rlb[1]:__rub[1], __rlb[2]:__rub[2]],
            )
        end

        new_imgs[:, :, ileft] .= unrot(imresize(img_new, newimg_size; method=Constant())) # nearest neighbor "interpolation"

        new_targets[1, ileft] = tdata[ileft]
        new_targets[2, ileft] = tdata[iright]
	end
    
    return new_imgs, new_targets
end

function read_idx_bytes(path)
    bytes = GZip.open(path, "r") do io
        read(io)
    end
    return bytes
end

function read_and_parse(path)
    bytes = read_idx_bytes(path)
    header, data = parse_idx(bytes)
    return bytes, header, data
end

function generate_multi_data(
    fpath, tpath; features_outpath=nothing, targets_outpath=nothing, kwargs...
)      
    fbytes, fheader, fdata = read_and_parse(fpath)
    tbytes, theader, tdata = read_and_parse(tpath)

    # generate multi-feature image data
    new_imgs, new_targets = make_new_imgs(fheader.shape, fdata, tdata; kwargs...)

    foutpath = isnothing(features_outpath) ? default_multi_path(fpath) : features_outpath
    toutpath = isnothing(targets_outpath) ? default_multi_path(tpath) : targets_outpath

    GZip.open(foutpath, "w") do io
        # the new images have same size, so simply reuse old header:
        write(io, fbytes[1:16])
        write(io, to_idx_bytes(new_imgs))
    end

    new_targets_bytes = to_idx_bytes(new_targets)
    GZip.open(toutpath, "w") do io
        write(io, 0x00)
        write(io, 0x00)
        write(io, 0x08)     # indicate UInt8 data
        write(io, 0x02)     # two dimensions
        # copy number of items
        write(io, tbytes[5:8])
        # now there are two rows, however
        write(io, hton(Int32(2)))
        write(io, new_targets_bytes)
    end
    return new_imgs, new_targets, foutpath, toutpath
end

function default_multi_path(path)
    return string(path) * ".multi.gz"
end

struct MultiMNIST <: SupervisedDataset
    metadata::Dict{String, Any}
    split::Symbol
    features::Array{<:Any, 3}
    targets::Matrix{Int}
end

"""
    MultiMNIST(
        mnist_dat::MNIST;
        features_outpath=nothing, targets_outpath=nothing,
        llb::Union{Symbol, Nothing, NTuple{2, <:Integer}}=nothing, 
        rlb::Union{Symbol, Nothing, NTuple{2, <:Integer}}=nothing, 
    )

Read or create the MultiMNIST dataset corresponding to `mnist_dat`.
The features consist of 28x28 pixel images resulting from randomly overlapping to MNIST 
images, see [1]. The labels are two dimensional columns of an integer matrix, with one entry
for the left image and one entry for the right digit.

By default, the function first looks at `features_outpath` and `targets_outpath` for 
existing IDX data files. If either keyword argument value is `nothing`, then we append 
"multi.gz" to the paths in `mnist_dat.metadata`. 
If one of the paths does not point to a file or if `llb` or `rlb` is not nothing, then new 
IDX files are created before creating the `MultiMNIST` data object.

`llb` is a Tuple for the corner of the lower left image and `rlb` for the upper right image.
If either is not nothing, this triggers the creation of new IDX files.
Use a Symbol (e.g. `llb=:reset`) to reset existing files to default values (used in [1]).

[1] O. Sener and V. Koltun, “Multi-Task Learning as Multi-Objective Optimization,” 
    arXiv:1810.04650 [cs, stat], Jan. 2019, Accessed: Jan. 24, 2022. [Online].  
    Available: http://arxiv.org/abs/1810.04650
"""
function MultiMNIST(
    mnist_dat::MNIST; 
    features_outpath=nothing, targets_outpath=nothing,
    llb::Union{Symbol, Nothing, NTuple{2, <:Integer}}=nothing, 
    rlb::Union{Symbol, Nothing, NTuple{2, <:Integer}}=nothing,
    kwargs...
)
    fpath = mnist_dat.metadata["features_path"]
	tpath = mnist_dat.metadata["targets_path"]

    ftype = eltype(mnist_dat.features)
    ttype = eltype(mnist_dat.targets)
    
    multi_fpath = isnothing(features_outpath) ? default_multi_path(fpath) : features_outpath
    multi_tpath = isnothing(targets_outpath) ? default_multi_path(tpath) : targets_outpath

    if !(isfile(multi_fpath) && isfile(multi_tpath)) || !(isnothing(llb)) || !isnothing(rlb)
        new_imgs, new_targets, _, _ = generate_multi_data(
            fpath, tpath;
            features_outpath = multi_fpath,
            targets_outpath = multi_tpath,
            llb, rlb, kwargs...
        )
    else
        _, _, new_imgs = read_and_parse(multi_fpath)
        _, _, new_targets = read_and_parse(multi_tpath)
    end

    features = convert(Array{ftype}, new_imgs)
    targets = convert(Array{ttype}, new_targets)

    return MultiMNIST(
        Dict(
            "n_observations" => last(size(features)),
            "features_path" => multi_fpath,
            "targets_path" => multi_tpath
        ),
        mnist_dat.split,
        features,
        targets
    )
end

end