function findGS(af, pm, pa, seq; iters=700, minpol = 0.90, nmax = 500, minbet = 0.02)
    @extract seq : ctype strseq
    @extract pm : L H J lambda_o lambda_e muext muint N
    
    polar = -1.0
    err = Inf
    energy = Inf
    n=0
    beta = pa.beta
    incbet = 0.1
    while (polar < minpol && n<nmax && incbet>minbet)

        af_old = deepcopy(af)
        pa.beta = beta
        @show beta        
        
        err = BpAlignGpu.test_sweep!(iters,af,pm,pa)
        xn, maxbel = BpAlignGpu.solmaxbel(af)
        seqsol = BpAlignGpu.convert_soltosequence!(xn, strseq, N, L)
        
        energy_old = energy
        energy = BpAlignGpu.compute_cost_function(J, H, seqsol[2], L, ctype, lambda_o, lambda_e, muext, muint)

        polar_old = polar
        polar = mean(maxbel)
        @show polar, energy, err
        if ( (polar_old > polar) && (err > pa.tol) ) 
            println("decrease incbet: ", incbet/2, " ** polar_old: ", polar_old, ", polar: ", polar, " ** energy_old: ", energy_old, ", energy: ", energy)
            beta -= incbet
            af = deepcopy(af_old)
            polar = polar_old
            energy = energy_old
            incbet /= 2
        else    
            n += 1
        end
        beta += incbet
    end
    beta -= incbet
    xnsol, maxbel = BpAlignGpu.solmaxbel(af)
    c = BpAlignGpu.check_sr!(xnsol, L, N)
    check = sum(c)
    bel = Array(af.bpb.beliefs)
    U = BpAlignGpu.internal_nrj(af, pm, pa)
    Φ = BpAlignGpu.freeent(af, pm, pa)
    S = (Φ + beta*U)/L
    
    return beta, err, polar, energy, check, U, S, xnsol, bel
end

function findGS_betarange(af, pm, pa, seq; iters=700, betarange = 0.0:0.1:1.0, minpol = 0.90)
    @extract seq : ctype strseq
    @extract pm : L H J lambda_o lambda_e muext muint N
    
    err = Inf
    polar = 0.0
    for beta in betarange
        if polar < minpol
            @show beta
            pa.beta = beta
            err = BpAlignGpu.test_sweep!(iters,af,pm,pa)
            xn, maxbel = BpAlignGpu.solmaxbel(af)
            polar = mean(maxbel)
            seqsol = BpAlignGpu.convert_soltosequence!(xn, strseq, N, L)
            energy = BpAlignGpu.compute_cost_function(J, H, seqsol[2], L, ctype, lambda_o, lambda_e, muext, muint)
            @show polar, energy, err
        end
    end
    
    xnsol, maxbel = BpAlignGpu.solmaxbel(af)
    seqsol = BpAlignGpu.convert_soltosequence!(xnsol, strseq, N, L)
        
    energy = BpAlignGpu.compute_cost_function(J, H, seqsol[2], L, ctype, lambda_o, lambda_e, muext, muint)

    polar = mean(maxbel)
    
    c = BpAlignGpu.check_sr!(xnsol, L, N)
    check = sum(c)
    
    bel = Array(af.bpb.beliefs)
    U = BpAlignGpu.internal_nrj(af, pm, pa)
    Φ = BpAlignGpu.freeent(af, pm, pa)
    S = (Φ + pa.beta*U)/L
    
    return pa.beta, err, polar, energy, check, U, S, xnsol, bel
end
