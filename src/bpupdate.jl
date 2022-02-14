ϕ(Δn, lambda_oi, lambda_ei) = (Δn > 0)*(lambda_oi + lambda_ei*(Δn - 1.0))



function χsr(nim1,xim1,ni,xi,N,lambda_oi::T,lambda_ei::T) where T <: AbstractFloat
    myone = T(1)
    myzero= T(0)
    Δn = ni - nim1 - 1
    if xim1 == xi == 1 
        return nim1 == ni ? myone : myzero
    elseif xim1==2 && xi ==1
        if ni == N + 1 
            return myone
        else
            return ni == nim1 ? myone : myzero
        end
    elseif xim1 == 1 && xi == 2
        if 0 <= nim1 < ni < N+1
            return exp(-ϕ( Δn,lambda_oi,lambda_ei) * (nim1 > 0))
        else
            return myzero
        end
    else
        if 0 < nim1 < ni < N+1
            return exp(- ϕ(Δn,lambda_oi,lambda_ei))
        else
            return myzero
        end
    end
    error("the end of the world has come")
end


function χin(n,x,N,T::DataType)
    myzero = T(0)
    myone = T(1)
    if x==1
        return n==0 ? myone : myzero 
    else
        return 0<n<N+1 ? myone : myzero
    end
end


function update_f!(af::AllFields)
    @extract af:lrf bpb bpm
    @extract lrf:f
    @extract bpb:conditional
    @extract bpm: Jseq

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

function update_hF(af::AllFields, pm::ParamModel)
    @extract af : lrf bpm
    @extract lrf : g
    @extract bpm : Jseq F hF lambda_e lambda_o
    @extract pm : N

    @tullio hF[ni, xi, i] = χsr(nim1, xim1, ni, xi, N, lambda_o[i], lambda_e[i]) * exp(Jseq[nim1, xim1, ni, xi, i-1, i] + g[nim1, xim1, ni, xi, i-1]) * F[nim1, xim1, i-1]
end

mumask(mint,muext,n) = 0<n<N+1 ? mint : muext  

function update_F(af::AllFields, pm::ParamModel)
    @extract af:lrf bpm
    @extract lrf:f
    @extract bpm:Hseq F hF
    @extract pm:muint muext
    T = typeof(muext)

    @tullio F[ni, xi, i] = hF[ni, xi, i] * exp(Hseq[ni, xi, i] - (1 - xi) * mumask(muint, muext, ni) + f[ni, xi, i])
    @tullio F[ni, xi, 1] = χin(ni,xi,N,T) * exp(Hseq[ni, xi, 1] - (1 - xi) * mumask(muint, muext, ni) + f[ni, xi, 1])
end