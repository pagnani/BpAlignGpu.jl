module BpAlignGpu

using Tullio, CUDA, LoopVectorization, CUDAKernels, KernelAbstractions
using ExtractMacro: @extract
#export crea_instance_tensor_long, test_instance_tensor_long!
export BPMessages, BPBeliefs, LongRangeFields, AllFields, Seq, ParamModel, ParamAlgo

include("types.jl")
include("tullio_functions.jl")
include("utils.jl")
end
