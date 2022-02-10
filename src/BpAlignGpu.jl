module BpAlignGpu

<<<<<<< HEAD
using Tullio, CUDA, LoopVectorization, CUDAKernels, KernelAbstractions
=======
using Tullio, LoopVectorization, CUDA, CUDAKernels, KernelAbstractions, Random, LinearAlgebra

include("update_longrange.jl")
include("tullio_functions.jl")
>>>>>>> 1c923a725c55897fa09f068329bf236d2f5145ee

#export crea_instance_tensor_long, test_instance_tensor_long!
export BPMessages, BPBeliefs, LongRangeFields, AllFields
#include("garbage.jl")
include("types.jl")
end
