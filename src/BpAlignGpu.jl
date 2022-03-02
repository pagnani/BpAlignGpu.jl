module BpAlignGpu

using FastaIO # needed for dataset.jl
using OffsetArrays #only needed for function 'decodeposterior' copy-pasted from DCAlign-master (and not needed for BP updates)
using Tullio, CUDA, LoopVectorization, CUDAKernels, KernelAbstractions
CUDA.allowscalar(false)
using ExtractMacro: @extract
export BPMessages, BPBeliefs, LongRangeFields, AllFields, Seq, ParamModel, ParamAlgo

include("dataset.jl") #copy-paste of DCAlign-master
include("types.jl")
include("bpupdate.jl")
include("newfun_bpupdate.jl")
include("utils.jl")
include("computesol.jl")
end
