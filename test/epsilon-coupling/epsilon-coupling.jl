using Revise, Pkg
Pkg.activate("/home/louise/MSA/BpAlignGpu.jl")
using BpAlignGpu
using Plots, Statistics, DelimitedFiles, CUDA

function extract_data(namefile::String)
    data = readdlm(namefile);
    param = data[1,:]
    nsamp =param[1]
    L = param[3]

    inds = data[2,1:nsamp];
    res = data[3:end,:];

    xnsols = fill((0,0), L, nsamp)
    for ns in 1:nsamp
        for i=1:L
            x = res[ns,i]
            n = res[ns,L+i]
            xnsols[i,ns] = (x,n)
        end
    end

    return param, inds, xnsols
end

function main(args)
    index = parse(Int64, args[1])
    ϵ = parse(Float32, args[2])
    argdamp = parse(Float32, args[3])
    
    namefile = "groundstate_PF00684_n60_mf_viterbi.txt"
    params, inds, xnsols0 = extract_data(namefile);
    idx0 = inds[index]
    xn0 = xnsols0[:,index]
    
    @show index, idx0, ϵ, argdamp
#------------------------------------------#------------------------------------------#-----------------------------------------
    CUDA.device!(0)
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

    L=67;
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
    epscoupling=(true, T(ϵ), xn0)
    pa = ParamAlgo(damp, tol, tolnorm, tmax, upscheme, initcond, lr, beta, verbose, epscoupling)
#------------------------------------------#------------------------------------------#------------------------------------------
    bpm = BPMessages(seq, pm, pa)
    bpb = BPBeliefs(N, L)
    lrf = LongRangeFields(N, L)
    af = AllFields(bpm, bpb, lrf)
#------------------------------------------#------------------------------------------#------------------------------------------
    ##find ground state
    iters = 800
    minpol = 2.0
    betarange = [0.0, 0.1, 0.2, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4]#0.0:0.1:0.4
    @time beta_ϵ, err_ϵ, polar_ϵ, energy_ϵ, check_ϵ, U_ϵ, S_ϵ, xnsol_ϵ, bel_ϵ = BpAlignGpu.findGS_betarange(af, pm, pa, seq; iters=iters, betarange = betarange, minpol = minpol)
#------------------------------------------#------------------------------------------#------------------------------------------
    #decimation using Viterbi
    xnsol_vit, p_vit = BpAlignGpu.viterbi_decoding(af, pm)
    seqsol_vit = BpAlignGpu.convert_soltosequence!(xnsol_vit, seq.strseq, N, L)
    energy_vit = BpAlignGpu.compute_cost_function(pm.J, pm.H, seqsol_vit[2], pm.L, seq.ctype, pm.lambda_o, pm.lambda_e, pm.muext, pm.muint)
    c = BpAlignGpu.check_sr!(xnsol_vit, L, N)
    if sum(c) > 0
        println("problem during viterbi: new check=", sum(c))
        exit(0)
    end
    
    Hdist = sum(xnsol_vit .!= xn0)/L
    @show index idx0 ϵ Hdist energy_vit U_ϵ S_ϵ polar_ϵ beta_ϵ err_ϵ check_ϵ
#------------------------------------------#------------------------------------------#------------------------------------------
    paramrun = [fam, L, M, delta, pa.damp, pa.tol, pa.tolnorm, pa.initcond, pa.lr, iters, betarange]
    file_param = "param_epsilon_coupling.txt"
    open(file_param, "a") do io
        writedlm(io, [paramrun])
    end    

    results = [index idx0 ϵ Hdist energy_vit U_ϵ S_ϵ polar_ϵ beta_ϵ err_ϵ check_ϵ]
    file_data = "data_epsilon_coupling.txt"
    open(file_data, "a") do io
        writedlm(io, [results])
    end    

    x0 = [x[1] for x in xnsol_vit]
    x1 = [x[2] for x in xnsol_vit]
    xc = vcat(x0, x1)
    file_solutions = "solutions_epsilon_coupling.txt"
    open(file_solutions, "a") do io
        writedlm(io, [xc])
    end
    
    
    println("\n")
    return nothing
end

main(ARGS)