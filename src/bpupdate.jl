@inline ϕ(Δn, lambda_oi, lambda_ei) = (Δn > 0) * (lambda_oi + lambda_ei * (Δn - 1))

function χsr(nim1, xim1, ni, xi, N, lambda_oi::T, lambda_ei::T, beta::T) where {T<:AbstractFloat}
    myone = T(1.0)
    myzero = T(0.0)
    Δn = ni - nim1 - 1
    if xim1 == xi == 1
        return nim1 == ni ? myone : myzero
    elseif xim1 == 2 && xi == 1
        if ni == N + 2
            return myone
        else
            return ni == nim1 ? myone : myzero
        end
    elseif xim1 == 1 && xi == 2
        if 1 <= nim1 < ni < N + 2
            return exp(-beta*ϕ(Δn, lambda_oi, lambda_ei) * (nim1 > 1))
        else
            return myzero
        end
    else
        if 1 < nim1 < ni < N + 2
            return exp(-beta*ϕ(Δn, lambda_oi, lambda_ei))
        else
            return myzero
        end
    end
    error("the end of the world has come")
end

@inline function χin(n, x, N)
    if x == 1
        return n == 1 ? true : false
    else
        return 1 < n < N + 2 ? true : false
    end
end

@inline function χend(n, x, N)
    if x == 1
        return n == N + 2 ? true : false
    else
        return 1 < n < N + 2 ? true : false
    end
end

@inline mumask(muint, muext, n, N) = 1 < n < N + 2 ? muint : muext

function update_hF!(af::AllFields, pm::ParamModel, pa::ParamAlgo)
    @extract af : lrf bpm
    @extract lrf : g
    @extract bpm : Jseq F hF lambda_e lambda_o scra
    @extract pm : N
    @extract pa : damp beta

    fill!(scra, 1.0)
    @tullio scra[ni, xi, i+1] = χsr(nim1, xim1, ni, xi, N, lambda_o[i+1], lambda_e[i+1], beta) * exp(beta*(Jseq[nim1, xim1, ni, xi, i, i+1] + g[nim1, xim1, ni, xi, i])) * F[nim1, xim1, i]
    normalize_3tensor!(scra)
    @tullio hF[ni, xi, i] = damp * hF[ni, xi, i] + (1-damp) * scra[ni, xi, i]
    normalize_3tensor!(hF)
    synchronize()
    return nothing
end

function update_hB!(af::AllFields, pm::ParamModel, pa::ParamAlgo)
    @extract af : lrf bpm
    @extract lrf : g
    @extract bpm : Jseq B hB lambda_e lambda_o scra
    @extract pm : N
    @extract pa : damp beta

    fill!(scra, 1.0)
    @tullio scra[ni, xi, i] = χsr(ni, xi, nip1, xip1, N, lambda_o[i+1], lambda_e[i+1], beta) * exp(beta*(Jseq[ni, xi, nip1, xip1, i, i+1] + g[ni, xi, nip1, xip1, i])) * B[nip1, xip1, i+1]
    scra[1, 2, :] .= 0.0
    scra[N+2, 2, :] .= 0.0
    normalize_3tensor!(scra)
    @tullio hB[ni, xi, i] = damp * hB[ni, xi, i] + (1 - damp) * scra[ni, xi, i]
    normalize_3tensor!(hB)
    synchronize()
    return nothing
end

function update_F!(af::AllFields, pm::ParamModel, pa::ParamAlgo)
    @extract af : lrf bpm
    @extract lrf : f
    @extract bpm : Hseq F hF scra
    @extract pm : muint muext N
    @extract pa : damp beta

    fill!(scra, 1.0)
    @tullio scra[ni, xi, i] = ((i > 1) ? hF[ni, xi, i] : χin(ni, xi, N)) * exp(beta*(Hseq[ni, xi, i] - (2 - xi) * mumask(muint, muext, ni, N) + f[ni, xi, i])) grad = false
    normalize_3tensor!(scra)
    @tullio F[ni, xi, i] = damp * F[ni, xi, i] + (1 - damp) * scra[ni, xi, i]
    normalize_3tensor!(F)
    synchronize()
    return nothing
end

function update_B!(af::AllFields, pm::ParamModel, pa::ParamAlgo)
    @extract af : lrf bpm
    @extract lrf : f
    @extract bpm : Hseq B hB scra
    @extract pm : muint muext N L
    @extract pa : damp beta

    fill!(scra, 1.0)
    @tullio scra[ni, xi, i] = ((i < L) ? hB[ni, xi, i] : χend(ni, xi, N)) * exp(beta*(Hseq[ni, xi, i] - (2 - xi) * mumask(muint, muext, ni, N) + f[ni, xi, i])) grad = false
    normalize_3tensor!(scra)
    @tullio B[ni, xi, i] = damp * B[ni, xi, i] + (1 - damp) * scra[ni, xi, i]
    normalize_3tensor!(B)
    synchronize()
    return nothing
end

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

function update_fold!(af::AllFields)
    @extract af:lrf bpb bpm
    @extract lrf:f
    @extract bpb:conditional
    @extract bpm:Jseq

    @tullio scra[ni, xi, nl, xl, i, l] := Jseq[ni, xi, nj, xj, i, j] * conditional[nj, xj, nl, xl, j, l] * (j > l)
    @tullio f[nl, xl, l] = -conditional[ni, xi, nl, xl, i, l] * scra[ni, xi, nl, xl, i, l] * (i < l)
    #@tullio f[nl, xl, l] = -conditional[ni, xi, nl, xl, i, l] * Jseq[ni, xi, nj, xj, i, j] * conditional[nj, xj, nl, xl, j, l] * (i < l) * (j > l)
    synchronize()
    return nothing
end

function update_f!(af::AllFields)
    @extract af:lrf bpb bpm
    @extract lrf:f
    @extract bpb:conditional
    @extract bpm:Jseq
    np1 = size(conditional, 1)
    L = size(conditional, 6)

    mask = similar(conditional)
    @tullio mask[ni, xi, nj, xj, i, j] = (i < j) (ni in 1:np1, xi in 1:2, nj in 1:np1, xj in 1:2, i in 1:L, j in 1:L)
    mask = reshape(permutedims(mask, (1, 2, 5, 3, 4, 6)), L * 2 * np1, L * 2 * np1)
    J = reshape(permutedims(Jseq, (1, 2, 5, 3, 4, 6)), L * 2 * np1, L * 2 * np1)
    cond = reshape(permutedims(conditional, (1, 2, 5, 3, 4, 6)), L * 2 * np1, L * 2 * np1)
        
    f .= reshape(diag(- (mask .* cond') * (J * (cond .* mask))), np1, 2, L)
    #CUDA.@time @tullio fscra[nl, xl, l] := -conditional[ni, xi, nl, xl, i, l] * Jseq[ni, xi, nj, xj, i, j] * conditional[nj, xj, nl, xl, j, l] * (i < l) * (j > l)
    synchronize()
    return nothing
end

function update_gold!(af::AllFields)
    @extract af:lrf bpb bpm
    @extract lrf:g
    @extract bpb:conditional
    @extract bpm:Jseq
    
    @tullio scra[nl, xl, nj, xj, j, l] := conditional[ni, xi, nl, xl, i, l] * Jseq[ni, xi, nj, xj, i, j] * ((i <= l) * (j > l) * (j > i + 1))
    @tullio g[nl, xl, nl1, xl1, l] = scra[nl, xl, nj, xj, j, l] * conditional[nj, xj, nl1, xl1, j, l+1] * (j > l)
    synchronize()
    return nothing
end

function update_g!(af::AllFields)
    @extract af:lrf bpb bpm
    @extract lrf:g
    @extract bpb:conditional
    @extract bpm:Jseq

    np1 = size(conditional, 1)
    L = size(conditional, 6)
    maskL = similar(conditional)
    maskC = similar(conditional)
    @tullio maskL[ni, xi, nj, xj, i, j] = (i <= j) (ni in 1:np1, xi in 1:2, nj in 1:np1, xj in 1:2, i in 1:L, j in 1:L)
    @tullio maskC[ni, xi, nj, xj, i, j] = (j > i + 1) (ni in 1:np1, xi in 1:2, nj in 1:np1, xj in 1:2, i in 1:L, j in 1:L)
    #@tullio maskR[ni, xi, nj, xj, i, j] = (i > j-1) (ni in 1:np1, xi in 1:2, nj in 1:np1, xj in 1:2, i in 1:L, j in 1:L)


    maskL = reshape(permutedims(maskL, (1, 2, 5, 3, 4, 6)), L * 2 * np1, L * 2 * np1)
    maskC = reshape(permutedims(maskC, (1, 2, 5, 3, 4, 6)), L * 2 * np1, L * 2 * np1)
    #maskR = reshape(permutedims(maskR, (1, 2, 5, 3, 4, 6)), L * 2 * np1, L * 2 * np1)
    J = reshape(permutedims(Jseq, (1, 2, 5, 3, 4, 6)), L * 2 * np1, L * 2 * np1)
    cond = reshape(permutedims(conditional, (1, 2, 5, 3, 4, 6)), L * 2 * np1, L * 2 * np1)
    scra = permutedims(reshape((maskL .* cond)' * ((J .* maskC) * (cond .* maskL')), np1, 2, L,np1, 2, L),(1,2,4,5,3,6))
    # @show size(scra)
    # res = CUDA.zeros(np1, 2, np1, 2, L)


    for i in 1:L-1
        g[:, :, :, :, i] .= scra[:, :, :, :, i, i+1]
    end

    #@tullio g[nl, xl, nl1, xl1, l] = conditional[ni, xi, nl, xl, i, l] * Jseq[ni, xi, nj, xj, i, j] * conditional[nj, xj, nl1, xl1, j, l+1] * ((i <= l) * (j > l) * (j > i + 1))
    synchronize()

    return nothing
end

#(normalize_3tensor!(ten::AbstractArray{T,3}) where T<:AbstractFloat) = ten .= ten ./ sum(ten, dims = (1, 2))
#(normalize_5tensor!(ten::AbstractArray{T,5}) where T<:AbstractFloat) = ten .= ten ./ sum(ten, dims = (1, 2, 3, 4))

function normalize_3tensor!(ten::AbstractArray{T,3}) where T<:AbstractFloat
    tolnorm=1f-20
    mynorm = sum(ten, dims = (1, 2))
    
    if all(mynorm .<= tolnorm) 
        println("normalization T3: mynorm .<= tolnorm")
    elseif any(isinf.(mynorm))
        println("normalization T3: Inf")
    elseif any(isnan.(mynorm))
        println("normalization T3: NaN")
    end
    ten .= ten ./ mynorm
end

function normalize_5tensor!(ten::AbstractArray{T,5}) where T<:AbstractFloat
    tolnorm=1f-20
    mynorm = sum(ten, dims = (1, 2, 3, 4))
    
    if all(mynorm .<= tolnorm) 
        println("normalization T5: mynorm .<= tolnorm")
    elseif any(isinf.(mynorm))
        println("normalization T5: Inf")
    elseif any(isnan.(mynorm))
        println("normalization T5: NaN")
    end
    ten .= ten ./ mynorm
end

function one_bp_sweep!(af::AllFields, pm::ParamModel, pa::ParamAlgo)
    @extract pa : lr
    
    CUDA.@time update_F!(af, pm, pa)
    CUDA.@time update_hF!(af, pm, pa)
    CUDA.@time update_B!(af, pm, pa)
    CUDA.@time update_hB!(af, pm, pa)
    CUDA.@time update_beliefs!(af, pm, pa)
    CUDA.@time update_jointchain!(af, pm, pa)
    if lr == :sce
        CUDA.@time update_conditional_chain!(af, pa)
        CUDA.@time update_conditional_all!(af, pm)
        CUDA.@time update_f!(af)
        CUDA.@time update_g!(af)
    end
end

function test_sweep!(n,af,pm,pa)
    @extract af : bpb
    @extract bpb : beliefs_old
    @extract pa : tol
    
    for t in 1:n
        beliefs_old .= af.bpb.beliefs
        one_bp_sweep!(af, pm, pa)
        err = maximum(abs.(beliefs_old .- af.bpb.beliefs))
        println("t=", t, "\t err=", err)
        if err < tol
            println("converged: err=", err, ", tol=", tol)
            return nothing
        end
        flush(stdout)
    end
    return nothing
end
