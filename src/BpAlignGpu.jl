module BpAlignGpu

using Tullio, CUDA, LoopVectorization, CUDAKernels, KernelAbstractions

#export crea_instance_tensor_long, test_instance_tensor_long!
export BPMessages, BPBeliefs, LongRangeFields, AllFields
#include("garbage.jl")
include("types.jl")
end
