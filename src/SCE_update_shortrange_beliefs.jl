function SCE_update_joint_chain!(i, beta, tolnorm, F, B, g, lambda_o, lambda_e, N, elts_Jseq)

    joint_chain = fill(0.0, 2N+4, 2N+4)

    #(x_i,n_i)=(0,n_i) and (x_{i+1},n_{i+1})=(0,n_{i+1})
    for y=1:N+2
        joint_chain[y,y] = exp( beta*elts_Jseq[y,y,i,i+1] + beta*g[y,y,i] )*F[y,i]*B[y,i+1]
    end

    #(x_i,n_i)=(1,n_i) and (x_{i+1},n_{i+1})=(0,n_{i+1})
    for n=0:N
        joint_chain[N+3+n, n+1] = exp( beta*elts_Jseq[N+3+n,n+1,i,i+1] + beta*g[N+3+n,n+1,i] )*F[N+3+n,i]*B[n+1,i+1]
    end
    for n=1:N
        joint_chain[N+3+n, N+2] = exp( beta*elts_Jseq[N+3+n,N+2,i,i+1] + beta*g[N+3+n,N+2,i] )*F[N+3+n,i]*B[N+2,i+1]
    end
    
    #(x_i,n_i)=(0,n_i) and (x_{i+1},n_{i+1})=(1,n_{i+1})
    for n=1:N
        joint_chain[1,N+3+n] = exp( beta*elts_Jseq[1,N+3+n,i,i+1] + beta*g[1,N+3+n,i] )*F[1,i]*B[N+3+n,i+1]
    end
    for ni=1:N
        for ni1 = ni+1:N
            inscost = insertioncost(ni1-ni-1, i+1, lambda_o, lambda_e)
            joint_chain[ni+1, N+3+ni1] = exp( -beta*inscost + beta*elts_Jseq[ni+1,N+3+ni1,i,i+1] + beta*g[ni+1,N+3+ni1,i] )*F[ni+1,i]*B[N+3+ni1,i+1]
        end
    end
    
    #(x_i,n_i)=(1,n_i) and (x_{i+1},n_{i+1})=(1,n_{i+1})
    for ni=1:N
        for ni1=ni+1:N
            inscost = insertioncost(ni1-ni-1, i+1, lambda_o, lambda_e)
            joint_chain[N+3+ni, N+3+ni1] = exp( -beta*inscost + beta*elts_Jseq[N+3+ni,N+3+ni1,i,i+1] + beta*g[N+3+ni,N+3+ni1,i] )*F[N+3+ni,i]*B[N+3+ni1,i+1]
        end
    end
    return joint_chain
end

function SCE_update_conditional_chain!(i, conditional, joint_chain, tolnorm, N)
    #P_{i,i+1}(y_i|y_{i+1})
    for yi1=1:2N+4
        S = sum(joint_chain[:,yi1, i])
        if S < tolnorm
            #println("conditional[:, $yi1, $i, $i+1]: normalization=$S")
            conditional[:, yi1, i, i+1] .= 0.0
        else
            for yi=1:2N+4
                conditional[yi, yi1, i, i+1] = joint_chain[yi, yi1, i]/S
            end
        end
    end
    
    #P_{i+1,i}(y_{i+1}|y_i)
    for yi=1:2N+4
        S = sum(joint_chain[yi, :, i])
        if S < tolnorm
            #println("conditional[:, $yi, $i+1, $i]: normalization=$S")
            conditional[:, yi, i+1, i] .=0.0
        else
            for yi1=1:2N+4
                conditional[yi1, yi, i+1, i] = joint_chain[yi, yi1, i]/S
            end
        end
    end
    sum(isnan.(conditional[:,:,i,i+1])) > 0 && println("NaNs in sr conditional[:,:,$i,$(i+1)]")
    sum(isnan.(conditional[:,:,i+1,i])) > 0 && println("NaNs in sr conditional[:,:,$(i+1),$i]")
    return conditional
end


