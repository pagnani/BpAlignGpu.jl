insertioncost(Δn,i, lambda_o, lambda_e) = (Δn > 0)*(lambda_o[i] + lambda_e[i]*(Δn - 1.0))

#update hF[:,i]
function SCE_update_hF!(i, F, damp, beta, tolnorm, g, lambda_o, lambda_e, N, elts_Jseq)
    new_M = fill(0.0, 2N+4)

    fill!(new_M, 0.0)

    #forbidden states (x_i,n_i)=(1,0) and (x_i,n_i)=(1,N+1)
    new_M[N+3]=0.0
    new_M[2N+4]=0.0 
        
    #for (x_i,n_i)=(0,n), n=0,...,N
    for ni=0:N
        new_M[ni+1] = exp( beta*elts_Jseq[ni+1, ni+1, i-1,i] + beta*g[ni+1, ni+1, i-1] ) * F[ni+1, i-1] 
        new_M[ni+1] += exp( beta*elts_Jseq[N+3+ni, ni+1, i-1,i] + beta*g[N+3+ni, ni+1, i-1]) * F[N+3+ni, i-1]
    end
    
    # for (x_i,n_i)=(0,N+1)
    new_M[N+2] = exp( beta*elts_Jseq[N+2, N+2, i-1,i] + beta*g[N+2, N+2, i-1]) * F[N+2, i-1]
    for n=1:N
        new_M[N+2] += exp( beta*elts_Jseq[N+3+n, N+2, i-1,i] + beta*g[N+3+n, N+2, i-1]) * F[N+3+n, i-1]
    end
    
    # for y=N+4 (i.e. (x_i,n_i)=(1,1)), only one term
    new_M[N+4] = exp( beta*elts_Jseq[1,N+4,i-1,i] + beta*g[1,N+4,i-1])*F[1,i-1] 
    # for (x_i,n_i)=(1,n), n=2,...,N
    for ni=2:N
        new_M[N+3+ni] = exp( beta*elts_Jseq[1,N+3+ni,i-1,i] + beta*g[1,N+3+ni,i-1])*F[1,i-1]
        for n=1:ni-1
            inscost = insertioncost(ni-n-1, i, lambda_o, lambda_e)
            new_M[N+3+ni] += exp( -beta*inscost + beta*elts_Jseq[n+1,N+3+ni,i-1,i] + beta*g[n+1,N+3+ni,i-1] )*F[n+1, i-1]
            new_M[N+3+ni] += exp( -beta*inscost + beta*elts_Jseq[N+3+n,N+3+ni,i-1,i] + beta*g[N+3+n,N+3+ni,i-1] )*F[N+3+n, i-1]
        end
    end
    return new_M
end

function SCE_update_F!(i, damp, beta, tolnorm, hF, f, muext, muint, N, elts_Hseq)
    new_M = fill(0.0, 2N+4)
    
    if i==1
        new_M[:].=0.0
        new_M[1] = exp( beta*elts_Hseq[1, 1] + beta*f[1, 1] - beta*muext)
        for y=N+4:2N+3
            new_M[y] = exp( beta*elts_Hseq[y, 1] + beta*f[y, 1])
        end
    else
        new_M[1] = exp( beta*elts_Hseq[1,i] - beta*muext + beta*f[1,i] )*hF[1,i]
        for y=2:N+1
            new_M[y] = exp( beta*elts_Hseq[y,i] - beta*muint + beta*f[y,i] )*hF[y,i]
        end
        new_M[N+2] = exp( beta*elts_Hseq[N+2,i] - beta*muext + beta*f[N+2,i] )*hF[N+2,i]

        new_M[N+3]=0.0
        for y=N+4:2N+3
            new_M[y] = exp( beta*elts_Hseq[y, i] + beta*f[y, i] )*hF[y, i]
        end
        new_M[2N+4]=0.0
    end

    return new_M
end

#update hB[:,i]
function SCE_update_hB!(i, damp, beta, tolnorm, B, g, lambda_o, lambda_e, N, elts_Jseq)
    new_M = fill(0.0, 2N+4)
    
    #forbidden states (x_i,n_i)=(1,0) and (x_i,n_i)=(1,N+1)
    new_M[N+3]=0.0
    new_M[2N+4]=0.0     
    
    # for (x_i,n_i)=(0,0)
    new_M[1] = exp( beta*elts_Jseq[1,1,i,i+1] + beta*g[1,1,i] )*B[1,i+1]
    for n=1:N
        new_M[1] += exp( beta*elts_Jseq[1,N+3+n,i,i+1] + beta*g[1,N+3+n,i] )*B[N+3+n,i+1]
    end
    
    # xi=0, ni=1:N-1 (i.e. y=2:N)
    for ni=1:N-1
        new_M[ni+1] = exp( beta*elts_Jseq[ni+1,ni+1,i,i+1] + beta*g[ni+1,ni+1,i] )*B[ni+1,i+1]
        for n=ni+1:N
            inscost = insertioncost(n-ni-1, i+1, lambda_o, lambda_e)
            new_M[ni+1] += exp( -beta*inscost + beta*elts_Jseq[ni+1,N+3+n,i,i+1] + beta*g[ni+1,N+3+n,i])*B[N+3+n,i+1]
        end
    end   
    
    # x=0, n=N (i.e. y=N+1)
    new_M[N+1] = exp( beta*elts_Jseq[N+1,N+1,i,i+1] + beta*g[N+1,N+1,i])*B[N+1,i+1]
    # x=0, n=N+1 (i.e. y=N+2)
    new_M[N+2] = exp( beta*elts_Jseq[N+2, N+2, i, i+1] + beta*g[N+2, N+2, i])* B[N+2, i+1]

    # x=1, n=1:N-1
    for ni=1:N-1
        new_M[N+3+ni] = exp( beta*elts_Jseq[N+3+ni, ni+1, i, i+1] + beta*g[N+3+ni, ni+1, i])*B[ni+1,i+1]
        new_M[N+3+ni] += exp( beta*elts_Jseq[N+3+ni, N+2, i, i+1] + beta*g[N+3+ni, N+2, i])*B[N+2,i+1]
        for n=ni+1:N
            inscost = insertioncost(n-ni-1, i+1, lambda_o, lambda_e)
            new_M[N+3+ni] += exp( -beta*inscost + beta*elts_Jseq[N+3+ni, N+3+n, i, i+1] + beta*g[N+3+ni, N+3+n, i])*B[N+3+n,i+1]
        end
    end
    # x=1, n=N
    new_M[2N+3] = exp( beta*elts_Jseq[2N+3, N+1, i, i+1] + beta*g[2N+3, N+1, i] )*B[N+1,i+1] 
    new_M[2N+3] += exp( beta*elts_Jseq[2N+3, N+2, i, i+1] + beta*g[2N+3, N+2, i])*B[N+2,i+1]
    
    return new_M
end

function SCE_update_B!(i, damp, beta, tolnorm, hB, f, muext, muint, N, L, elts_Hseq)
    new_M = fill(0.0, 2N+4)
    
    if i == L
        new_M[:].=0.0
        new_M[N+2] = exp( beta*elts_Hseq[N+2, L] + beta*f[N+2, L] - beta*muext )
        for y=N+4:2N+3
            new_M[y] = exp( beta*elts_Hseq[y, L] + beta*f[y, L] )
        end
    else
        new_M[1] = exp( beta*elts_Hseq[1, i] - beta*muext + beta*f[1, i] )*hB[1, i]
        for y=2:N+1
            new_M[y] = exp( beta*elts_Hseq[y, i] - beta*muint + beta*f[y, i] )*hB[y, i]
        end
        new_M[N+2] = exp( beta*elts_Hseq[N+2, i] - beta*muext + beta*f[N+2, i] )*hB[N+2, i]

        new_M[N+3]=0.0
        for y=N+4:2N+3
            new_M[y] = exp( beta*elts_Hseq[y, i] + beta*f[y, i] )*hB[y, i]
        end
        new_M[2N+4]=0.0
    end

    return new_M
end


