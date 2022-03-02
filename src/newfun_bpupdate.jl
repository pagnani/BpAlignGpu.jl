function update_beliefs!(af::AllFields, pm::ParamModel, pa::ParamAlgo)
    @extract af : lrf bpm bpb
    @extract lrf : f
    @extract bpm : hF hB Hseq
    @extract bpb : beliefs
    @extract pm : N L muint muext
    @extract pa : beta
    
    @tullio beliefs[ni, xi, i] = ((i> 1) ? hF[ni, xi, i] : χin(ni, xi, N)) * ((i<L) ? hB[ni, xi, i] : χend(ni, xi, N)) * exp(beta*(Hseq[ni, xi, i] - (2 - xi) * mumask(muint, muext, ni, N) + f[ni, xi, i])) grad = false
    normalize_3tensor!(beliefs)
    return nothing
end

function update_jointchain!(af::AllFields, pm::ParamModel, pa::ParamAlgo)
    @extract af : lrf bpm bpb
    @extract lrf : g
    @extract bpb : joint_chain
    @extract bpm : Jseq F B lambda_e lambda_o
    @extract pm : N
    @extract pa : beta

    @tullio joint_chain[ni, xi, nip1, xip1, i] = χsr(ni, xi, nip1, xip1, N, lambda_o[i+1], lambda_e[i+1], beta) * exp(beta*(Jseq[ni, xi, nip1, xip1, i, i+1] + g[ni, xi, nip1, xip1, i])) * F[ni, xi, i] * B[nip1, xip1, i+1]
    normalize_5tensor!(joint_chain)
    return nothing
end

function update_conditional_chain!(af::AllFields, pa::ParamAlgo)
    @extract af : bpb
    @extract bpb : joint_chain conditional
    @extract pa : tolnorm
    
    v = sum(joint_chain, dims=(1,2))
    w = sum(joint_chain, dims=(3,4))
    @tullio conditional[ni, xi, nip1, xip1, i, i+1] = (v[1,1, nip1, xip1, i] > tolnorm) ? joint_chain[ni, xi, nip1, xip1, i] /v[1,1, nip1, xip1, i] : 0.0
    @tullio conditional[nip1, xip1, ni, xi, i+1, i] = (w[ni,xi, 1, 1, i] > tolnorm) ? joint_chain[ni, xi, nip1, xip1, i] /w[ni,xi, 1, 1, i] : 0.0
    return nothing
end


function update_conditional_all!(af::AllFields, pm::ParamModel)
    @extract af : bpb
    @extract bpb : conditional
    @extract pm : L N
    
    for i=1:L-2
        for j = i+1:L-1
            #@tullio C[ni, xi, nj, xj, i, j+1] = conditional[ni, xi, n, x, i, j] * conditional[n, x, nj, xj, j,j+1]
            C1 = view(conditional, :, :, :, :, i, j)
            C2 = view(conditional, :, :, :, :, j, j + 1)
            conditional[:, :, :, :, i, j+1] .= reshape(reshape(C1, 2(N + 2), 2(N + 2)) * reshape(C2, 2(N + 2), 2(N + 2)), N + 2, 2, N + 2, 2)
        end
    end    
    for j=1:L-2
        for i = j+1:L-1
            #@tullio C[ni, xi, nj, xj, i+1, j] = conditional[ni, xi, n, x, i+1, i] * conditional[n, x, nj, xj, i, j]
            C1 = view(conditional, :, :, :, :, i + 1, i)
            C2 = view(conditional, :, :, :, :, i, j)
            conditional[:, :, :, :, i+1, j] .= reshape(reshape(C1, 2(N + 2), 2(N + 2)) * reshape(C2, 2(N + 2), 2(N + 2)), N + 2, 2, N + 2, 2)
        end
    end
    return nothing
end