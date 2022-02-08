module BpAlignGpu

using Tullio, LoopVectorization, CUDA, CUDAKernels, KernelAbstractions, Random, LinearAlgebra

include("update_longrange.jl")
include("tullio_functions.jl")

end
