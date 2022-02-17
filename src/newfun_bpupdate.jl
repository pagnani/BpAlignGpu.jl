function update_jointchain!(af::AllFields, pm::ParamModel)
    @extract af : lrf bpm bpb
    @extract lrf : g
    @extract bpb : joint_chain
    @extract bpm : Jseq F B lambda_e lambda_o
    @extract pm : N
    @tullio joint_chain[ni, xi, nip1, xip1, i] = χsr(ni, xi, nip1, xip1, N, lambda_o[i+1], lambda_e[i+1]) * exp(Jseq[ni, xi, nip1, xip1, i, i+1] + g[ni, xi, nip1, xip1, i]) * F[ni, xi, i] * B[nip1, xip1, i+1]
    return nothing
end

function update_conditional_chain!(af::AllFields, pa::ParamAlgo)
    @extract af : bpb
    @extract bpb : joint_chain conditional
    @extract pa : tolnorm
    
    v = sum(af.bpb.joint_chain, dims=(1,2))
    w = sum(af.bpb.joint_chain, dims=(3,4))
    @tullio conditional[ni, xi, nip1, xip1, i, i+1] = (v[1,1, nip1, xip1, i] > tolnorm) ? joint_chain[ni, xi, nip1, xip1, i] /v[1,1, nip1, xip1, i] : 0.0
    @tullio conditional[nip1, xip1, ni, xi, i+1, i] = (w[ni,xi, 1, 1, i] > tolnorm) ? joint_chain[ni, xi, nip1, xip1, i] /w[ni,xi, 1, 1, i] : 0.0
    return nothing
end

function update_conditional_all!(af::AllFields, pm::ParamModel)
    @extract af : bpb
    @extract bpb : conditional
    @extract pm : L N
    
    CR = fill(0.0, N+2, 2, N+2, 2, L, L) |> cu
    CL = fill(0.0, N+2, 2, N+2, 2, L, L) |> cu

    @tullio CR[ni, xi, nj, xj, i, j+1] = (1<=i<=L-2 && i+1<=j<L ) ? (conditional[ni, xi, n, x, i, j] * conditional[n, x, nj, xj, j,j+1]) : 0.0

    @tullio CL[ni, xi, nj, xj, i+1, j] = (1<=j<=L-2 && j+1<=i<L ) ? (conditional[ni, xi, n, x, i+1,i] * conditional[n, x, nj, xj, i,j]) : 0.0

    return CR, CL
end

function update_conditional_all_forloop!(af::AllFields, pm::ParamModel)
    @extract af : bpb
    @extract bpb : conditional
    @extract pm : L N
    
    C = fill(0.0, N+2, 2, N+2, 2, L, L) |> cu
    for i=1:L-2
        for j=i+1:L-1
            @tullio C[ni, xi, nj, xj, i, j+1] = conditional[ni, xi, n, x, i, j] * conditional[n, x, nj, xj, j,j+1]
        end
    end    
    for j=1:L-2
        for i=j+1:L-1
            @tullio C[ni, xi, nj, xj, i+1, j] = conditional[ni, xi, n, x, i+1,i] * conditional[n, x, nj, xj, i,j]
        end
    end

    return C
end