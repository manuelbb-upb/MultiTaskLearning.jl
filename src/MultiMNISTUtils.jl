module MultiMNISTUtils

export MultiMNIST

import MLDatasets: MNIST, SupervisedDataset, bytes_to_type, MNISTReader
import Interpolations: Constant
import GZip
import ImageTransformations: imresize
import Random: Xoshiro

"""
    global_coordinates(tlc; local_img_size=(28,28), tmp_size=(36,36))

Compute actual top left and bottom right corner coordinates of an image with specified top
left corner `tlc` (negative values allowed) and image size `local_img_size` in a global 
canvas of size `tmp_size.
"""
function global_coordinates(
    tlc;
    local_img_size=(28, 28), tmp_size=(36, 36)
)
    
    TLC = (1,1)                     # coordinates of global top left corner
    BRC = tmp_size                  # coordinates of global bottom right corner
    brc = tlc .+ local_img_size .- 1    # coordinates of subimage bottom right corner

    # project back into global constraints:
    _tlc = min.(BRC, max.(TLC, tlc))
    _brc = min.(BRC, max.(TLC, brc))
    return _tlc, _brc
end

"""
    local_coordinates(tlc, brc, orig_size)

Consider an image `img` with size `orig_size`, a slice of which should be placed in `canvas`
such that the slice starts at `canvas[tlc...]` and its bottom right corner is
`img[orig_size...] = canvas[brc...]`.
Return image space coordinate tuples `_tlc` and `_brc` such that
`img[_tlc[1]:_brc[1], _tlc[2]:_brc[2]] == canvas[tlc[1]:brc[1], tlc[2]:brc[2]]`.
"""
function local_coordinates(tlc, brc, orig_size)
    # actual width of subimage
    w = brc .- tlc .+ 1
    _tlc = orig_size .- w .+ 1
    _brc = orig_size
    return _tlc, _brc
end

function overlap(ltlc, lbrc, rtlc, rbrc, lloc_tlc, lloc_brc, rloc_tlc, rloc_brc)
    diag = lbrc .- rtlc

    lov_tlc = lloc_tlc
    lov_brc = lloc_brc

    rov_tlc = rloc_tlc
    rov_brc = rloc_brc

    has_overlap = all(diag .> 0)    # there is overlap, in the global canvas its top left corner is rtlc, and its bottom right corner is lbrc
    if has_overlap
        lov_tlc = lloc_brc .- diag
        rov_brc = rloc_tlc .+ diag
    end
    
    return has_overlap, lov_tlc, lov_brc, rov_tlc, rov_brc
end

function corner_view(arr, tlc, brc)
    return view(arr, tlc[1]:brc[1], tlc[2]:brc[2])
end

#=
Image coordinates: Top-left corner is [1,1]. 
=#
function make_new_imgs(fdata, tdata;
    top_left_corner_left::Union{Symbol, Nothing, NTuple{2,<:Integer}}=nothing,
    top_left_corner_right::Union{Symbol, Nothing, NTuple{2,<:Integer}}=nothing,
    newimg_size::Union{Nothing, NTuple{2,<:Integer}}=nothing,
    tmp_size::Union{Nothing, NTuple{2,<:Integer}}=nothing,
)

    rng = Xoshiro(31415)      # reproducibly choose RHS image
    
    dim = ndims(fdata)
    @assert dim <= 3

    permtuple = (2,1,3:dim...)
    invpremtuple = invperm(permtuple)
    pfdata = permutedims(fdata, permtuple)  # switch rows and columns, parsed IDX data is transposed
    s1, s2, len = size(pfdata)
    
    # top_left_corner_left = (1,1), top_left_corner_right=(7,7), tmp_size=(36,36) is used by Sener & Koltun
    tlc_l = if isnothing(top_left_corner_left) || top_left_corner_left isa Symbol
        (1,1)
    else
        top_left_corner_left
    end
    tlc_r = if isnothing(top_left_corner_right) || top_left_corner_right isa Symbol
        (7,7)
    else
        top_left_corner_right
    end
    if isnothing(newimg_size)
        newimg_size = (s1, s2)
    end
    if isnothing(tmp_size) || tmp_size isa Symbol
        tmp_size = newimg_size .+ 8
    end
    if tlc_l[2] >= tlc_r[2]
        # left image must be left...
        _tlc_l = tlc_l
        tlc_l = tlc_r
        tlc_r = _tlc_l
    end

    # compute coordinates of left and right subimage, taking negative entries into account
    ltlc, lbrc = global_coordinates(tlc_l; local_img_size=(s1, s2), tmp_size)
    rtlc, rbrc = global_coordinates(tlc_r; local_img_size=(s1, s2), tmp_size)

    lloc_tlc, lloc_brc = local_coordinates(ltlc, lbrc, (s1, s2))
    rloc_tlc, rloc_brc = local_coordinates(rtlc, rbrc, (s1, s2))

    has_overlap, lov_tlc, lov_brc, rov_tlc, rov_brc = overlap(ltlc, lbrc, rtlc, rbrc, lloc_tlc, lloc_brc, rloc_tlc, rloc_brc)

    #=
    @info "Left:  tlc = $(ltlc), brc = $(lbrc), loc_tlc = $(lloc_tlc), loc_brc = $(lloc_brc), ov_tlc=$(lov_tlc), ov_brc=$(lov_brc)."
    @info "Right: tlc = $(rtlc), brc = $(rbrc), loc_tlc = $(rloc_tlc), loc_brc = $(rloc_brc), ov_tlc=$(rov_tlc), ov_brc=$(rov_brc)."
    @show rtlc[1]:lbrc[1]
    @show rtlc[2]:lbrc[2]
    =#
    
    new_imgs = similar(pfdata)
    new_targets = similar(tdata, (2, len))

    for ileft = 1:len
		iright = rand(rng, 1:len)    # choose random index for right feature img
		img_left = pfdata[:,:,ileft]
		img_right = pfdata[:,:,iright]
		img_new = zeros(eltype(img_left), tmp_size)
	    
        corner_view(img_new, ltlc, lbrc) .= corner_view(img_left, lloc_tlc, lloc_brc)
        corner_view(img_new, rtlc, rbrc) .= corner_view(img_left, rloc_tlc, rloc_brc)
        
        # overlap: maximum value
        if has_overlap
            corner_view(img_new, rtlc, lbrc) .= max.(
                corner_view(img_left, lov_tlc, lov_brc),
                corner_view(img_right, rov_tlc, rov_brc),
            )
        end

        new_imgs[:, :, ileft] .= imresize(img_new, newimg_size; method=Constant()) # nearest neighbor "interpolation"

        new_targets[1, ileft] = tdata[ileft]
        new_targets[2, ileft] = tdata[iright]
	end
    
    return permutedims(new_imgs, invpremtuple), new_targets
end

#=================== IDX Utility functions ===============================#

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

const INV_DATATYPE_DICT = Base.ImmutableDict(
    UInt8 => 0x08,
    Int8 => 0x09,
    Int16 => 0x0B,
    Int32 => 0x0C,
    Float32 => 0x0D,
    Float64 => 0x0E
)

function read_idx(path)
    bytes = GZip.open(path, "r") do io
        parse_idx_io(io)
    end
    return bytes
end

function parse_idx_io(io)
    seekstart(io)

    # first 2 bytes are zero
    @assert iszero(read(io, UInt8))
    @assert iszero(read(io, UInt8))

    # next 2 bytes encode type and dimensionality
    dtype = DATTYPE_DICT[read(io, UInt8)]
    dims = Int(read(io, UInt8))

    # shape is given by Int32 values
    shape = zeros(Int32, dims)
    for i=1:dims
        shape[i] = ntoh(only(reinterpret(Int32, read(io, 4))))
    end

    # rest is data, but row-major (Julia reads column-major, hence the `reverse`)
    data_mat = Array{dtype}(undef, reverse(shape)...)
    read!(io, data_mat)

    return (dims=dims, shape=shape, dtype=dtype), ntoh.(data_mat)
end

function to_idx_bytes(arr)
    return reshape(reinterpret(UInt8, hton.(arr)), :)
end

function write_idx(outpath, arr)
    if haskey(INV_DATATYPE_DICT, eltype(arr))
        dtype = INV_DATATYPE_DICT[eltype(arr)]
        _arr = arr
    else
        if eltype(arr) <: Real
            dtype = Float64
            _arr = Array{Float64}(arr)
        else
            error("array has unsupported data type.")
        end
    end
    
    dims = UInt8(ndims(arr))
    shape = size(arr)

    try
        GZip.open(outpath, "w") do io
            write(io, 0x00)
            write(io, 0x00)
            write(io, dtype)
            write(io, dims)
            for len in reverse(shape)
                write(io, hton(Int32(len)))
            end
            write(io, to_idx_bytes(_arr))
        end
        return outpath
    catch
        @warn "Could not save newly generated data in IDX file at\n$(outpath)"
        return nothing
    end
end

function write_idx_files(features_array, targets_array, features_outpath, targets_outpath)
    @assert last(size(features_array)) == last(size(targets_array))

    fo = write_idx(features_outpath, features_array)
    to = write_idx(targets_outpath, targets_array)
    return fo, to
end

function generate_multi_data(
    fpath, tpath; kwargs...
)      
    _, fdata = read_idx(fpath)
    _, tdata = read_idx(tpath)

    # generate multi-feature image data
    return make_new_imgs(fdata, tdata; kwargs...)
end

function default_multi_path(path)
    return string(path) * ".multi.gz"
end

struct MultiMNIST <: SupervisedDataset
    metadata::Dict{String, Any}
    split::Symbol
    features::Array{<:Any, 3}
    targets::Matrix{UInt8}      # MNIST uses Int64, a bit excessive...
end

function MultiMNIST(FT::Type, split::Symbol, features_path, targets_path)
    features = bytes_to_type(FT, last(read_idx(features_path)))
    targets = Matrix{UInt8}(last(read_idx(targets_path)))

    return MultiMNIST(
        Dict(
            "n_observations" => last(size(features)),
            "features_path" => features_path,
            "targets_path" => targets_path
        ),
        split,
        features,
        targets
    )
end

function MultiMNIST(mnist;
    force_recreate::Bool=false, features_outpath=nothing, targets_outpath=nothing, kwargs...
)
    fpath = mnist.metadata["features_path"]
	tpath = mnist.metadata["targets_path"]
    multi_fpath = isnothing(features_outpath) ? default_multi_path(fpath) : features_outpath
    multi_tpath = isnothing(targets_outpath) ? default_multi_path(tpath) : targets_outpath

    if !force_recreate && isfile(multi_fpath) && isfile(multi_tpath)
        return MultiMNIST(eltype(mnist.features), mnist.split, multi_fpath, multi_tpath)
    end

    multi_features, multi_targets = generate_multi_data(fpath, tpath; kwargs...)
    
    multi_fpath, multi_tpath = write_idx_files(multi_features, multi_targets, multi_fpath, multi_tpath)
    
    return MultiMNIST(
        Dict(
            "n_observations" => last(size(multi_features)),
            "features_path" => multi_fpath,
            "targets_path" => multi_tpath
        ),
        mnist.split,
        bytes_to_type(eltype(mnist.features), multi_features),
        multi_targets
    )
end

end