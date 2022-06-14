using Revise, Pkg
Pkg.activate("/home/louise/MSA/BpAlignGpu.jl")
using BpAlignGpu
using Plots, Statistics, DelimitedFiles, CUDA

function main(args)
    idx0 = parse(Int64, args[1])
    argminpol = parse(Float64, args[2])
    argdamp = parse(Float32, args[3])
    @show idx0, argminpol, argdamp
    
    CUDA.device!(1)
    #CUDA.allowscalar(false)
#------------------------------------------#------------------------------------------#-----------------------------------------
    T = Float32
    q = 21
    ctype=Symbol("amino")
    typel=Symbol("bm");

    muext = 0.00
    muint = 2.50;
#------------------------------------------#------------------------------------------#-----------------------------------------
    fam = "PF00684"
    open("/home/louise/MSA/Data/test/PF00684/")
    L = 67; 

    J, H = BpAlignGpu.read_parameters("/home/louise/MSA/Data/test/PF00684/Parameters_bm_PF00684seed_potts.dat", q, L, gap=0, typel=typel);

    delta = 50;
    al = BpAlignGpu.enveloptoalign( "/home/louise/MSA/Data/test/PF00684/Test_PF00684.full", "/home/louise/MSA/Data/test/PF00684/Test_PF00684.fasta", "/home/louise/MSA/Data/test/PF00684/Test_PF00684.ins", delta = delta, ctype = ctype);
    M = length(al)

    Lambda_all = readdlm("/home/louise/MSA/Data/test/PF00684/Lambda_PF00684.dat")
    lambda_o = Lambda_all[:,1];
    lambda_e = Lambda_all[:,2];
#------------------------------------------#------------------------------------------#-----------------------------------------
    seq = BpAlignGpu.Seq(al[idx0][3], al[idx0][2], ctype)
    N = length(al[idx0][2])
    pm = ParamModel{T}(N, L, q, muint, muext, lambda_o, lambda_e, H, J)
#------------------------------------------#------------------------------------------#-----------------------------------------
    #build paramalgo
    damp=T(argdamp)
    tol=T(1e-3)
    tolnorm=T(1e-10)
    tmax=10
    upscheme=:sequential 
    initcond=:random 
    lr=:sce  
    beta=T(0.0)
    verbose=false
    ϵ = -0.0
    xnsol = fill((0, 0), L);
    epscoupling=(false, T(ϵ), xnsol)
    pa = ParamAlgo(damp, tol, tolnorm, tmax, upscheme, initcond, lr, beta, verbose, epscoupling)
#------------------------------------------#------------------------------------------#------------------------------------------
    bpm = BPMessages(seq, pm, pa)
    bpb = BPBeliefs(N, L)
    lrf = LongRangeFields(N, L)
    af = AllFields(bpm, bpb, lrf)
#------------------------------------------#------------------------------------------#------------------------------------------
    ##find ground state
    iters = 800
    minpol = argminpol
    betarange = 0.1:0.1:1.0
    @time beta_ϵ, err_ϵ, polar_ϵ, energy_ϵ, check_ϵ, U_ϵ, S_ϵ, xnsol_ϵ, bel_ϵ = BpAlignGpu.findGS_betarange(af, pm, pa, seq; iters=iters, betarange = betarange, minpol = minpol)

    @show beta_ϵ, err_ϵ, polar_ϵ, energy_ϵ, check_ϵ, U_ϵ, S_ϵ
#------------------------------------------#------------------------------------------#------------------------------------------
    P = fill(fill(0.0, 0:1,0:N+1), L)
    BpAlignGpu.reshape_T3(bel_ϵ,P)
    s, P = BpAlignGpu.decimate_post(seq, P, L, q, N, false);
    seqsol_nucl = BpAlignGpu.decodeposterior(P, seq.strseq)
    sat = BpAlignGpu.check_assignment(P,false,N)
    if sat == false
        println("problem during nucleation")
        exit(0)
    end
    xnsol_nucl = BpAlignGpu.convertseqtoxnsol(seqsol_nucl, pm)
    energy_nucl = BpAlignGpu.compute_cost_function(pm.J, pm.H, seqsol_nucl[2], pm.L, seq.ctype, pm.lambda_o, pm.lambda_e, pm.muext, pm.muint)

    #decimation using Viterbi
    xnsol_vit, p_vit = BpAlignGpu.viterbi_decoding(af, pm)
    seqsol_vit = BpAlignGpu.convert_soltosequence!(xnsol_vit, seq.strseq, N, L)
    energy_vit = BpAlignGpu.compute_cost_function(pm.J, pm.H, seqsol_vit[2], pm.L, seq.ctype, pm.lambda_o, pm.lambda_e, pm.muext, pm.muint)
    c = BpAlignGpu.check_sr!(xnsol_vit, L, N)
    if sum(c) > 0
        println("problem during viterbi: new check=", sum(c))
        exit(0)
    end

    @show energy_ϵ, energy_nucl, energy_vit
#------------------------------------------#------------------------------------------#------------------------------------------
    paramrun = [40, fam, L, M, delta, pa.damp, pa.tol, pa.tolnorm, pa.initcond, pa.lr, iters, minpol]
    namefile = "inds.txt"
    open(namefile, "a") do io
        writedlm(io, [paramrun])
    end    
#------------------------------------------#------------------------------------------#------------------------------------------
    namefile = "run_GS_"*String(fam)*"_n40_"*String(pa.lr)*"_maxP.txt"
    x0 = [x[1] for x in xnsol_ϵ]
    x1 = [x[2] for x in xnsol_ϵ]
    xc = vcat(x0, x1)
    open(namefile, "a") do io
        writedlm(io, [xc])
    end

    namefile = "run_GS_"*String(fam)*"_n40_"*String(pa.lr)*"_nucleation.txt"
    x0 = [x[1] for x in xnsol_nucl]
    x1 = [x[2] for x in xnsol_nucl]
    xc = vcat(x0, x1)
    open(namefile, "a") do io
        writedlm(io, [xc])
    end

    namefile = "run_GS_"*String(fam)*"_n40_"*String(pa.lr)*"_viterbi.txt"
    x0 = [x[1] for x in xnsol_vit]
    x1 = [x[2] for x in xnsol_vit]
    xc = vcat(x0, x1)
    open(namefile, "a") do io
        writedlm(io, [xc])
    end
#------------------------------------------#------------------------------------------#------------------------------------------

    println("\n")
    return nothing
end

main(ARGS)