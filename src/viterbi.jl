function viterbi_decoding(af, pm)
    @extract pm : N L
    @extract af : bpb
    @extract bpb : conditional joint_chain
    
    pr = zeros(N+2,2,L-1)
    sl = fill((0,0), N+2, 2, L-1)
    solviterbi = fill((0,0), L)
    
    P12 = Array(joint_chain[:,:,:,:,1])
    for ni=1:N+2
        for xi=1:2
            pr[ni,xi,1], sl[ni,xi,1] = findmax( view(P12,:,:,ni,xi) )
        end
    end
    
    for i=2:L-1
        Ci = Array(conditional[:,:,:,:,i+1,i])
        for ni=1:N+2
            for xi=1:2
                pr[ni,xi,i], sl[ni,xi,i] = findmax(pr[:, :, i-1].*Ci[ni,xi,:,:])
            end
        end
    end
    
    pviterbi, solL = findmax(pr[:,:,L-1])    
    solviterbi[L] = (solL[2]-1, solL[1]-1)
    
    for i=L-1:-1:1
        (xip1, nip1) = solviterbi[i+1]
        (ni, xi) = sl[nip1+1, xip1+1, i]
        solviterbi[i] = (xi-1, ni-1)
    end

    return solviterbi, pviterbi
end

function viterbi_sampling(af, pm)
    @extract pm : N L
    @extract af : bpb
    @extract bpb : beliefs conditional

    solsampled = fill((0,0), L)

    f = rand(1:L) # choose site to sample first
    Pf = Array(beliefs[:,:,f]);
    pf = reshape(Pf, 2(N+2),1)[:,1];
    wf = weights(pf)
    rf = sample(eachindex(pf), wf)
    if rf > N+2
        (xf,nf)=(1,rf-N-3)
    else
        (xf,nf)=(0,rf-1)
    end
    solsampled[f] = (xf,nf)
    
    (x,n)=(xf, nf)
    for i=f+1:L # sample site i knowing choice for site i-1
        Ci = Array(conditional[:,:,n+1,x+1,i,i-1])
        ci = reshape(Ci, 2(N+2), 1)[:,1]
        wi = weights(ci)
        ri = sample(eachindex(ci), wi)
        if ri > N+2
            (xi,ni)=(1,ri-N-3)
        else
            (xi,ni)=(0,ri-1)
        end
        solsampled[i] = (xi,ni)
        (x,n) = (xi,ni)
    end

    (x,n)=(xf, nf)
    for i=f-1:-1:1 # sample site i knowing choice for site i+1
        Ci = Array(conditional[:,:,n+1,x+1,i,i+1])
        ci = reshape(Ci, 2(N+2), 1)[:,1]
        wi = weights(ci)
        ri = sample(eachindex(ci), wi)
        if ri > N+2
            (xi,ni)=(1,ri-N-3)
        else
            (xi,ni)=(0,ri-1)
        end
        solsampled[i] = (xi,ni)
        (x,n) = (xi,ni)
    end

    return solsampled
end
