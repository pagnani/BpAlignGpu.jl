function crea_instance_matrix(N, L; T = Float32)
    A = rand(T, N * L, N * L) |> cu
    x = rand(T, N * L) |> cu
    return A, x
end
function crea_instance_tensor(N, L; T = Float32)
    A = rand(T, N, L, N, L) |> cu
    x = rand(T, N, L) |> cu
    return A, x
end

function crea_instance_tensor_long(N, L; T = Float32)
    A = rand(T, N, 2, N, 2, L, L) |> cu
    x1 = rand(T, N, 2) |> cu
    x2 = rand(T, N, 2) |> cu
    y = rand(L, L) |> cu
    return A, x1, x2, y
end

function crea_instance_J_cond(Q, L; T=Float32, gpu=true)
    gpufun = gpu ? cu : identity
    cond = rand(T, Q, Q, L, L) |> gpufun
    J = rand(T, Q, Q, L, L) |> gpufun
    return cond, J
end

test_instance_matrix(A, x) = @tullio y[a] := A[a, b] * x[b] grad = false
test_instance_tensor(A, x) = @tullio y[a, b] := A[a, b, c, d] * x[c, d] grad = false
test_instance_tensor_aliased(A, x, idxN, idxL) = @tullio y[idxN[a], idxL[b]] := A[a, b, c, d] * x[c, d] grad = false
test_matrix_prod(A, x) = A * x
test_prod_mat!(y, A, x) = @tullio y[a] := A[b, a] * x[b] * (a >= b) grad = false
test_mul_mat!(y, A, x) = mul!(y, A, x)
test_instance_tensor_long!(y, A, F1, F2) = @tullio y[xl, nl, l] = A[xi, ni, xj, nj, i, j] * F1[xl, nl, xj, nj, l, j]*F2[xi, ni, xl, nl, i, l]

test_update_f(cond, J) = @tullio f[yl, l] := - cond[yi, yl, i, l] * J[yi, yj, i, j] * cond[yj, yl, j, l] * (i < l) * (j > l)
test_update_g(cond, J) = @tullio g[yl, yl1, l] := cond[yi, yl, i, l] * J[yi, yj, i, j] * cond[yj, yl1, j, l+1] * (i <= l) * (j > l) * (j > i+1)