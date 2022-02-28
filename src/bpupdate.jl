@inline ϕ(Δn, lambda_oi, lambda_ei) = (Δn > 0) * (lambda_oi + lambda_ei * (Δn - 1))

function χsr(nim1, xim1, ni, xi, N, lambda_oi::T, lambda_ei::T) where {T<:AbstractFloat}
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
            return exp(-ϕ(Δn, lambda_oi, lambda_ei) * (nim1 > 1))
        else
            return myzero
        end
    else
        if 1 < nim1 < ni < N + 2
            return exp(-ϕ(Δn, lambda_oi, lambda_ei))
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

function update_f2!(af::AllFields)
    @extract af:lrf bpb bpm
    @extract lrf:f
    @extract bpb:conditional
    @extract bpm:Jseq


    CUDA.@time @tullio scra[ni, xi, nl, xl, i, l] := Jseq[ni, xi, nj, xj, i, j] * conditional[nj, xj, nl, xl, j, l] * (j > l)
    CUDA.@time @tullio f[nl, xl, l] = -conditional[ni, xi, nl, xl, i, l] * scra[ni, xi, nl, xl, i, l] * (i < l)
    #@tullio f[nl, xl, l] = -conditional[ni, xi, nl, xl, i, l] * Jseq[ni, xi, nj, xj, i, j] * conditional[nj, xj, nl, xl, j, l] * (i < l) * (j > l)
    synchronize()
    return nothing
end


function update_f!(af::AllFields)
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



function update_fold!(af::AllFields)
    @extract af:lrf bpb bpm
    @extract lrf:f
    @extract bpb:conditional
    @extract bpm:Jseq

    @tullio f[nl, xl, l] = -conditional[ni, xi, nl, xl, i, l] * Jseq[ni, xi, nj, xj, i, j] * conditional[nj, xj, nl, xl, j, l] * (i < l) * (j > l)
    synchronize()
   
end

function update_g!(af::AllFields)
    @extract af:lrf bpb bpm
    @extract lrf:g
    @extract bpb:conditional
    @extract bpm:Jseq

    @tullio scra[nl, xl, nj, xj, j, l] := conditional[ni, xi, nl, xl, i, l] * Jseq[ni, xi, nj, xj, i, j] * ((i <= l) * (j > l) * (j > i + 1))
    @tullio g[nl, xl, nl1, xl1, l] = scra[nl, xl, nj, xj, j, l] * conditional[nj, xj, nl1, xl1, j, l+1] * (j > l) 
    synchronize()
    return nothing
end


function update_gold!(af::AllFields)
    @extract af:lrf bpb bpm
    @extract lrf:g
    @extract bpb:conditional
    @extract bpm:Jseq

    @tullio g[nl, xl, nl1, xl1, l] = conditional[ni, xi, nl, xl, i, l] *
                                     Jseq[ni, xi, nj, xj, i, j] * conditional[nj, xj, nl1, xl1, j, l+1] * ((i <= l) * (j > l) * (j > i + 1))
    synchronize()
    return nothing
end

function update_hF!(af::AllFields, pm::ParamModel, pa::ParamAlgo)
    @extract af : lrf bpm
    @extract lrf : g
    @extract bpm : Jseq F hF lambda_e lambda_o scra
    @extract pm : N
    @extract pa : damp

    fill!(scra, 0.0)
    @tullio scra[ni, xi, i+1] = χsr(nim1, xim1, ni, xi, N, lambda_o[i+1], lambda_e[i+1]) * exp(Jseq[nim1, xim1, ni, xi, i, i+1] + g[nim1, xim1, ni, xi, i]) * F[nim1, xim1, i]
    normalize_3tensor!(scra)
    @tullio hF[ni, xi, i] = damp * hF[ni, xi, i] + (1-damp) * scra[ni, xi, i]
    normalize_3tensor!(hF)
    synchronize()
    return nothing
end

function update_hB!(af::AllFields, pm::ParamModel, pa::ParamAlgo)
    @extract af:lrf bpm
    @extract lrf:g
    @extract bpm:Jseq B hB lambda_e lambda_o scra
    @extract pm:N
    @extract pa:damp

    fill!(scra, 0.0)
    @tullio scra[ni, xi, i] = χsr(ni, xi, nip1, xip1, N, lambda_o[i+1], lambda_e[i+1]) * exp(Jseq[ni, xi, nip1, xip1, i, i+1] + g[ni, xi, nip1, xip1, i]) * B[nip1, xip1, i+1]
    scra[1, 2, :] .= 0.0
    scra[N+2, 2, :] .= 0.0
    normalize_3tensor!(scra)
    @tullio hB[ni, xi, i] = damp * hB[ni, xi, i] + (1 - damp) * scra[ni, xi, i]
    normalize_3tensor!(hB)
    synchronize()
    return nothing
end

@inline mumask(muint, muext, n, N) = 1 < n < N + 2 ? muint : muext

function update_F!(af::AllFields, pm::ParamModel, pa::ParamAlgo)
    @extract af:lrf bpm
    @extract lrf:f
    @extract bpm:Hseq F hF scra
    @extract pm:muint muext N
    @extract pa:damp

    fill!(scra, 0.0)
    @tullio scra[ni, xi, i] = ((i > 1) ? hF[ni, xi, i] : χin(ni, xi, N)) * exp(Hseq[ni, xi, i] - (2 - xi) * mumask(muint, muext, ni, N) + f[ni, xi, i]) grad = false
    normalize_3tensor!(scra)
    @tullio F[ni, xi, i] = damp * F[ni, xi, i] + (1 - damp) * scra[ni, xi, i]
    normalize_3tensor!(F)
    synchronize()
    return nothing
end

function update_B!(af::AllFields, pm::ParamModel, pa::ParamAlgo)
    @extract af:lrf bpm
    @extract lrf:f
    @extract bpm:Hseq B hB scra
    @extract pm:muint muext N L
    @extract pa:damp

    fill!(scra, 0.0)
    @tullio scra[ni, xi, i] = ((i < L) ? hB[ni, xi, i] : χend(ni, xi, N)) * exp(Hseq[ni, xi, i] - (2 - xi) * mumask(muint, muext, ni, N) + f[ni, xi, i]) grad = false
    normalize_3tensor!(scra)
    @tullio B[ni, xi, i] = damp * B[ni, xi, i] + (1 - damp) * scra[ni, xi, i]
    normalize_3tensor!(B)
    synchronize()
    return nothing
end

(normalize_3tensor!(ten::AbstractArray{T,3}) where T<:AbstractFloat) = ten .= ten ./ sum(ten, dims = (1, 2))
(normalize_5tensor!(ten::AbstractArray{T,5}) where T<:AbstractFloat) = ten .= ten ./ sum(ten, dims = (1, 2, 3, 4))

function one_bp_sweep!(af::AllFields, pm::ParamModel, pa::ParamAlgo)
    CUDA.@time update_F!(af, pm, pa)
    CUDA.@time update_hF!(af, pm, pa)
    CUDA.@time update_B!(af, pm, pa)
    CUDA.@time update_hB!(af, pm, pa)
    CUDA.@time update_beliefs!(af, pm)
    CUDA.@time update_jointchain!(af, pm)
    CUDA.@time update_conditional_chain!(af, pa)
    CUDA.@time update_conditional_all!(af, pm)
    CUDA.@time update_f!(af)
    CUDA.@time update_g!(af)
end

function test_sweep!(n,af,pm,pa)
    @extract af : bpb
    @extract bpb : beliefs_old
    for t in 1:n
        beliefs_old .= af.bpb.beliefs
        one_bp_sweep!(af, pm, pa)
        err = maximum(abs.(beliefs_old .- af.bpb.beliefs))
        println("t=", t, "\t err=", err)
    end
    nothing
end