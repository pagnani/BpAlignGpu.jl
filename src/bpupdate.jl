ϕ(Δn, lambda_oi, lambda_ei) = (Δn > 0) * (lambda_oi + lambda_ei * (Δn - 1.0))



function χsr(nim1, xim1, ni, xi, N, lambda_oi::T, lambda_ei::T) where {T<:AbstractFloat}
    myone = T(1)
    myzero = T(0)
    Δn = ni - nim1 - 1
    if xim1 == xi == 1
        return nim1 == ni ? myone : myzero
    elseif xim1 == 2 && xi == 1
        if ni == N + 1
            return myone
        else
            return ni == nim1 ? myone : myzero
        end
    elseif xim1 == 1 && xi == 2
        if 0 <= nim1 < ni < N + 1
            return exp(-ϕ(Δn, lambda_oi, lambda_ei) * (nim1 > 0))
        else
            return myzero
        end
    else
        if 0 < nim1 < ni < N + 1
            return exp(-ϕ(Δn, lambda_oi, lambda_ei))
        else
            return myzero
        end
    end
    error("the end of the world has come")
end


@inline function χin(n, x, N)
    if x == 1
        return n == 0 ? true : false
    else
        return 0 < n < N + 1 ? true : false
    end
end

@inline function χend(n, x, N)
    if x == 1
        return n == N + 1 ? true : false
    else
        return 0 < n < N + 1 ? true : false
    end
end

function update_f!(af::AllFields)
    @extract af:lrf bpb bpm
    @extract lrf:f
    @extract bpb:conditional
    @extract bpm:Jseq

    @tullio f[nl, xl, l] = -conditional[ni, xi, nl, xl, i, l] * Jseq[ni, xi, nj, xj, i, j] * conditional[nj, xj, nl, xl, j, l] * (i < l) * (j > l)
    return nothing
end

function update_g!(af::AllFields)
    @extract af:lrf bpb bpm
    @extract lrf:g
    @extract bpb:conditional
    @extract bpm:Jseq

    @tullio g[nl, xl, nl1, xl1, l] = conditional[ni, xi, nl, xl, i, l] * Jseq[ni, xi, nj, xj, i, j] * conditional[nj, xj, nl1, xl1, j, l+1] * (i <= l) * (j > l) * (j > i + 1)
    return nothing
end

function update_hF!(af::AllFields, pm::ParamModel)
    @extract af:lrf bpm
    @extract lrf:g
    @extract bpm:Jseq F hF lambda_e lambda_o
    @extract pm:N
    #println(typeof(lambda_e), " ", typeof(Jseq), " ",typeof(g), " ",typeof(F), " ",typeof(hF))
    @tullio hF[ni, xi, i+1] = χsr(nim1, xim1, ni, xi, N, lambda_o[i], lambda_e[i]) * exp(Jseq[nim1, xim1, ni, xi, i, i+1] + g[nim1, xim1, ni, xi, i]) * F[nim1, xim1, i]
end


@inline mumask(mint, muext, n, N) = 0 < n < N + 1 ? mint : muext

function update_F!(af::AllFields, pm::ParamModel)
    @extract af:lrf bpm
    @extract lrf:f
    @extract bpm:Hseq F hF
    @extract pm:muint muext N
    @tullio F[ni, xi, i] = hF[ni, xi, i] * exp(Hseq[ni, xi, i] - (1 - xi) * mumask(muint, muext, ni, N) + f[ni, xi, i]) grad = false
    @tullio F[ni, xi, 1] = χin(ni, xi, N) * exp(Hseq[ni, xi, 1] - (1 - xi) * mumask(muint, muext, ni, N) + f[ni, xi, 1]) grad = false
    return nothing
end


function update_B!(af::AllFields, pm::ParamModel)
    @extract af:lrf bpm
    @extract lrf:f
    @extract bpm:Hseq B hB
    @extract pm:muint muext N
    @tullio B[ni, xi, i] = hB[ni, xi, i] * exp(Hseq[ni, xi, i] - (1 - xi) * mumask(muint, muext, ni, N) + f[ni, xi, i]) grad = false
    @tullio B[ni, xi, L] = χend(ni, xi, N) * exp(Hseq[ni, xi, L] - (1 - xi) * mumask(muint, muext, ni, N) + f[ni, xi, L]) grad = false
    return nothing
end

function update_hB!(af::AllFields, pm::ParamModel, pa::ParamAlgo)
    @extract af:lrf bpm
    @extract lrf:g
    @extract bpm:Jseq B hB lambda_e lambda_o
    @extract pm:N
    @extract pa:damp
    #println(typeof(lambda_e), " ", typeof(Jseq), " ",typeof(g), " ",typeof(F), " ",typeof(hF))
    @tullio hB[ni, xi, i] = damp * hB[ni, xi, i] + (1 - damp) * χsr(nim1, xim1, ni, xi, N, lambda_o[i], lambda_e[i]) * exp(Jseq[nim1, xim1, ni, xi, i, i+1] + g[nim1, xim1, ni, xi, i]) * B[nim1, xim1, i+1]
end

(normalize_3tensor!(ten::AbstractArray{T,3}) where T<:AbstractFloat) = ten .= ten ./ sum(ten, dims = (1, 2))
(normalize_5tensor!(ten::AbstractArray{T,5}) where T<:AbstractFloat) = ten .= ten ./ sum(ten, dims = (1, 2, 3, 4))

function one_bp_sweep!(af::AllFields, pm::ParamModel, pa::ParamAlgo)
    update_F!(af, pm)
    update_hF!(af, pm)
    update_B!(af, pm)
    update_hB!(af, pm, pa)
    normalize_fields!(af)
end