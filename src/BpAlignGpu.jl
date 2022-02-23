module BpAlignGpu

using FastaIO # needed for dataset.jl
using Tullio, CUDA, LoopVectorization, CUDAKernels, KernelAbstractions
CUDA.allowscalar(false)
using ExtractMacro: @extract
#export crea_instance_tensor_long, test_instance_tensor_long!
export BPMessages, BPBeliefs, LongRangeFields, AllFields, Seq, ParamModel, ParamAlgo

include("dataset.jl") #copy-paste of DCAlign-master
include("types.jl")
include("bpupdate.jl")
include("newfun_bpupdate.jl")
include("utils.jl")
end
