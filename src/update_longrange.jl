function computediagprod!(result, A, B) #compute diag(A*B)
    result .= dot.(eachrow(A), eachcol(B))
end

function computediagprodtranspose!(result, A, B) #compute diag(A'*B)
    result .= dot.(eachcol(A), eachcol(B))
end

function update_f!(f, JP, tmpJP, tmpf, incf, conditional, elts_Jseq)
    L = size(conditional,3)
    Q = size(conditional,1)
    
    fill!(tmpf,0.0)
    fill!(JP,0.0)
    fill!(tmpJP,0.0)
    fill!(incf,0.0)
    
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
    
function update_g!(g, conditional, JP, tmpJP, tmpg, incg, elts_Jseq)
    L = size(conditional,3)
    Q = size(conditional,1)
    
    fill!(tmpg,0.0)
    fill!(tmpJP,0.0)
    fill!(JP,0.0)
    fill!(incg,0.0)

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

function update_lr_free_nrj(beliefs, conditional, diagJP, tmpdiagJP, elts_Jseq)
    L = size(conditional,3)
    Q = size(conditional,1)
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