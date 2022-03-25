function logZi(af::AllFields, pm::ParamModel, pa::ParamAlgo)
    @extract af : lrf bpm bpb
    @extract lrf : f
    @extract bpm : hF hB Hseq
    @extract bpb : beliefs
    @extract pm : N L muint muext
    @extract pa : beta
    
    @tullio beliefs[ni, xi, i] = ((i> 1) ? hF[ni, xi, i] : χin(ni, xi, N)) * ((i<L) ? hB[ni, xi, i] : χend(ni, xi, N)) * exp(beta*(Hseq[ni, xi, i] - (2 - xi) * mumask(muint, muext, ni, N) + f[ni, xi, i])) grad = false
    
    mynorm = sum(beliefs, dims = (1, 2))
    logZi = sum(log.(mynorm))
    normalize_3tensor!(beliefs)
    return logZi, log.(mynorm)
end

function logZa(af::AllFields, pm::ParamModel, pa::ParamAlgo)
    @extract af : lrf bpm bpb
    @extract lrf : g
    @extract bpb : joint_chain
    @extract bpm : Jseq F B lambda_e lambda_o
    @extract pm : N L
    @extract pa : beta

    @tullio joint_chain[ni, xi, nip1, xip1, i] = χsr(ni, xi, nip1, xip1, N, lambda_o[i+1], lambda_e[i+1], beta) * exp(beta*(Jseq[ni, xi, nip1, xip1, i, i+1] + g[ni, xi, nip1, xip1, i])) * F[ni, xi, i] * B[nip1, xip1, i+1]

    mynorm = sum(joint_chain, dims = (1, 2, 3, 4))
    logZa = sum(log.(mynorm[1:L-1]))
    normalize_5tensor!(joint_chain)
    return logZa, log.(mynorm)
end


function logZia(af::AllFields, pm::ParamModel)
    @extract af : bpm
    @extract bpm : F hF B hB scra
    @extract pm : N L

    #edge of type i,(i,i+1)
    @tullio scra[ni, xi, i] = F[ni, xi, i] * hB[ni, xi, i]
    mynorm = sum(scra, dims=(1,2))
    logZ1 = sum(log.(mynorm[1:L-1]))
    
    #edge of type (i,i+1), i+1
    @tullio scra[ni, xi, i] = B[ni, xi, i] * hF[ni, xi, i]
    mynorm = sum(scra, dims=(1,2))
    logZ2 = sum(log.(mynorm[2:L]))
    
    return logZia = logZ1 + logZ2  
end

function lr_freeen(af::AllFields, pm::ParamModel)
    @extract af : bpb bpm
    @extract bpb : conditional beliefs
    @extract bpm : Jseq
    @extract pm : L N
    
    mat = zeros( 2*(N+2), 2*(N+2) )
    res = 0.0
    for i=1:L-2
        for j=i+2:L
            C = view(conditional, :, :, :, :, j, i)
            J = view(Jseq, :, :, :, :, i, j)
            M = view(beliefs, :, :, i)
            mat = reshape( J, 2(N+2), 2(N+2)) * reshape(C, 2(N+2), 2(N+2))
            res += sum( diag(mat) .* reshape(M, 2(N+2)) )
        end
    end
    return res
end
