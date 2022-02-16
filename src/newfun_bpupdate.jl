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

    @tullio CR[ni, xi, njp1, xjp1, i, j+1] = (1<=i<=L-2 && j>=i+1 ) ? (conditional[ni, xi, nj, xj, i, j] * conditional[nj, xj, njp1, xjp1, j,j+1]) : 0.0
#    @tullio CR[ni, xi, nj, xj, i, j] = (1<=i<=L-2 && j>=i+2 && j>1) ? (conditional[ni, xi, njm1, xjm1, i, j-1] * conditional[njm1, xjm1, nj, xj, j-1,j]) : 0.0

    @tullio CL[nip1, xip1, nj, xj, i+1, j] = (1<=j<=L-2 && i>=j+1 ) ? (conditional[nip1, xip1, ni, xi, i+1,i] * conditional[ni, xi, nj, xj, i,j]) : 0.0
    #conditional = conditional .+ CR .+ CL
    return CR, CL
end