module BpAlignGpu

using FastaIO # needed for dataset.jl
using OffsetArrays #only needed for function 'decodeposterior' copy-pasted from DCAlign-master (and not needed for BP updates)
using Tullio, CUDA, LoopVectorization, CUDAKernels, KernelAbstractions, Statistics
CUDA.allowscalar(false)
using ExtractMacro: @extract
export BPMessages, BPBeliefs, LongRangeFields, AllFields, Seq, ParamModel, ParamAlgo
using LinearAlgebra: diag, mul!

include("dataset.jl") #copy-pasted from DCAlign-master
include("types.jl")
include("bpupdate.jl")
include("utils.jl")
include("computesol.jl")
include("free-energy.jl")
include("decimation.jl") #copy-pasted from DCAlign-master
include("functionsforanalysis.jl") #functions for epsilons-coupling analysis
end
