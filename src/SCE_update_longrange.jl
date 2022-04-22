function SCE_update_conditional_lr!(conditional)

    L= size(conditional, 3)

    @inbounds for j=1:L-2
        for i=j+2:L
            C1 = view(conditional,:,:,i,i-1)
            C2 = view(conditional,:,:,i-1,j)
            conditional[:,:,i,j] .= C1*C2
        end
    end    
    @inbounds for j=3:L
        for i=j-2:-1:1
            C1 = view(conditional,:,:,i,i+1)
            C2 = view(conditional,:,:,i+1,j)
            conditional[:,:,i,j] .= C1*C2
        end
    end
end

function computediagprod!(result, A, B) #compute diag(A*B)
    result .= dot.(eachrow(A), eachcol(B))
end

function computediagprodtranspose!(result, A, B) #compute diag(A'*B)
    result .= dot.(eachcol(A), eachcol(B))
end

function SCE_update_f!(f, conditional, elts_Jseq, N)
    JP = fill(0.0, 2N+4, 2N+4) #for f and g
    tmpJP = fill(0.0, 2N+4, 2N+4) #for f and g
    tmpf = fill(0.0, 2N+4) #for f
    incf = fill(0.0, 2N+4) #for f

    L = size(conditional,3)
    Q = size(conditional,1)
    
    
    @inbounds for l=2:L-1
        fill!(tmpf, 0.0)
        for i=1:l-1
            fill!(JP,0.0)
            fill!(incf, 0.0)
            Cil = view(conditional, :,:,i, l)
            for j=l+1:L
                Jij = view(elts_Jseq, :,:,i, j)
                Cjl = view(conditional, :,:,j, l)
                mul!(tmpJP,Jij,Cjl)
                JP .= JP .+ tmpJP 
            end
            computediagprodtranspose!(incf,Cil,JP)
            tmpf .= tmpf .- incf
        end
        f[:,l] .= tmpf
    end
end

function SCE_update_f_MF!(f, beliefs, J, N, L, intseq)
    q = size(J, 1)
    for i=1:L
        # case xᵢ = 1 (match)
        A0 = q
        @inbounds for nᵢ = 1:N
            Anᵢ = intseq[nᵢ]
            expscra = 0.0
            for j = 1:i-2
                for nⱼ = 0:nᵢ-1 # light cone constraint nⱼ < nᵢ
                    Anⱼ = nⱼ == 0 ? A0 : intseq[nⱼ]
                    expscra += J[Anⱼ, Anᵢ, j, i] * beliefs[N+3+nⱼ, j]
                    expscra += J[A0, Anᵢ, j, i] * beliefs[1+nⱼ, j]
                end
            end
            for j = i+2:L
                for nⱼ = nᵢ:N+1 # light cone constraint nⱼ > nᵢ
                    Anⱼ = nⱼ == N + 1 ? A0 : intseq[nⱼ]
                    expscra += nᵢ == nⱼ ? 0.0 : J[Anⱼ, Anᵢ, j, i] * beliefs[N+3+nⱼ, j]
                    expscra += J[A0, Anᵢ, j, i] * beliefs[1+nⱼ, j]
                end
            end
            f[N+3+nᵢ, i] = expscra
        end
        # case xᵢ = 0 (gap)
        @inbounds for nᵢ = 0:N+1
            expscra = 0.0
            for j = 1:i-2
                for nⱼ = 0:nᵢ
                    Anⱼ = (nⱼ == 0 || nⱼ == N + 1) ? A0 : intseq[nⱼ]
                    expscra += J[Anⱼ, A0, j, i] * beliefs[N+3+nⱼ, j]
                    expscra += J[A0, A0, j, i] * beliefs[1+nⱼ, j]
                end
            end
            for j = i+2:L
                for nⱼ = nᵢ:N+1
                    Anⱼ = (nⱼ == 0 || nⱼ == N + 1) ? A0 : intseq[nⱼ]
                    expscra += nᵢ == nⱼ ? 0.0 : J[Anⱼ, A0, j, i] * beliefs[N+3+nⱼ, j]
                    expscra += J[A0, A0, j, i] * beliefs[1+nⱼ, j]
                end
            end
            f[1+nᵢ, i] = expscra
        end
        #cannot match n = 0 pointer
        f[N+3, i] = -Inf
        f[2N+4, i] = -Inf
    end
    return nothing
end

function SCE_update_g!(g, conditional, elts_Jseq, N)
    JP = fill(0.0, 2N+4, 2N+4) #for f and g
    tmpJP = fill(0.0, 2N+4, 2N+4) #for f and g
    tmpg = fill(0.0, 2N+4, 2N+4) #for g
    incg = fill(0.0, 2N+4, 2N+4) #for g

    L = size(conditional,3)
    Q = size(conditional,1)
    
    @inbounds for l=1:L-1
        fill!(tmpg, 0.0)
        for i=1:l
            Cil = view(conditional, :,:,i, l)
            fill!(JP, 0.0)
            fill!(incg, 0.0)
            init_j = max(l+1,i+2) 
            for j=init_j:L
                Jij = view(elts_Jseq, :,:,i, j)
                Cjl1 = view(conditional, :,:,j, l+1)
                mul!(tmpJP,Jij,Cjl1)
                JP .= JP .+ tmpJP
            end
            mul!(incg, Cil',JP)
            tmpg .= tmpg .+ incg
        end
        g[:,:,l] .= tmpg
    end
end

function update_lr_free_nrj(beliefs, conditional, elts_Jseq)
    L = size(conditional,3)
    Q = size(conditional,1)

    tmpdiagJP = fill(0.0, Q)
    diagJP = fill(0.0, Q)

    lrfreeen = 0.0
    
    fill!(tmpdiagJP,0.0)
    fill!(diagJP,0.0)
    tmpres = 0.0
    @inbounds for i=1:L-2
        fill!(diagJP,0.0)
        Ci = view(beliefs, :,i)
        for j=i+2:L
            Cji = view(conditional,:,:,j,i)
            Jij = view(elts_Jseq,:,:,i,j)
            computediagprod!(tmpdiagJP,Jij,Cji)
            diagJP .= diagJP .+ tmpdiagJP
        end
        tmpres = dot(Ci,diagJP)
        lrfreeen += tmpres
    end
    lrfreeen
end
