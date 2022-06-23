using Revise, Pkg
Pkg.activate("/home/louise/MSA/BpAlignGpu.jl")
using BpAlignGpu
using Plots, Statistics, DelimitedFiles, CUDA
using ExtractMacro: @extract

function extract_data(namefile::String)
    data = readdlm(namefile);
    param = data[1,:]
    nsamp = param[1]
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

    return inds, xnsols
end
                
function find_sol(index, idx0, xn0, ϵ, seq, pm, T, damp, tol, tolnorm, initcond, lr, iters, betarange)
    @extract pm : N L
    
    #build paramalgo
    tmax=10
    upscheme=:sequential 
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
    minpol = 2.0
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
    return Hdist, energy_vit, U_ϵ, S_ϵ, polar_ϵ, beta_ϵ, err_ϵ, check_ϵ, xnsol_vit
end

function main(args)
    index = parse(Int64, args[1])
    
    namefile = "groundstate_PF00684_n60_mf_viterbi.txt"
    inds, xnsols0 = extract_data(namefile);
    idx0 = inds[index]
    xn0 = xnsols0[:,index]
    
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
    
    epsilons = [-0.0, -0.2, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0, -1.2, -1.3, -1.5]
    damp=T(0.2)
    tol=T(1e-3)
    tolnorm=T(1e-10)
    initcond=:random 
    lr=:sce  
    iters = 300
    betarange = 0.1:0.1:0.4
    
    results = zeros(length(epsilons), 11)
    xnsols_eps = fill(0, 2*L, length(epsilons));

    for (i,ϵ) in pairs(epsilons)
        @show index, idx0, ϵ, damp
        Hdist, energy_vit, U_ϵ, S_ϵ, polar_ϵ, beta_ϵ, err_ϵ, check_ϵ, xnsol_vit = find_sol(index, idx0, xn0, ϵ, seq, pm, T, damp, tol, tolnorm, initcond, lr, iters, betarange)
        println("\n")
        if err_ϵ > 5.0*tol
            println("---> this sequence does not converge <---")
            println("\n")
            exit(0)
        else
            results[i,:] = [index idx0 ϵ Hdist energy_vit U_ϵ S_ϵ polar_ϵ beta_ϵ err_ϵ check_ϵ]
            x0 = [x[1] for x in xnsol_vit]
            x1 = [x[2] for x in xnsol_vit]
            xc = vcat(x0, x1)
            xnsols_eps[:,i] = xc
        end
    end
#------------------------------------------#------------------------------------------#------------------------------------------
    paramrun = [fam, L, M, delta, damp, tol, tolnorm, initcond, lr, iters, betarange]
    file_param = "param_epsilon_coupling.txt"
    open(file_param, "a") do io
        writedlm(io, [paramrun])
    end    

    file_data = "data_epsilon_coupling.txt"
    for (i,ϵ) in pairs(epsilons)
        open(file_data, "a") do io
            writedlm(io, [results[i,:]])
        end    
    end

    file_solutions = "solutions_epsilon_coupling.txt"
    for (i,ϵ) in pairs(epsilons)
        open(file_solutions, "a") do io
            writedlm(io, [xnsols_eps[:,i]])
        end
    end
    println("\n")
    return nothing
end

main(ARGS)