function crea_instance_tensor_long(N, L; T = Float32)
    A = rand(T, N, 2, N, 2, L, L) |> cu
    F1 = rand(T, N, 2, N, 2, L, L) |> cu
    F2= rand(T, N, 2, N, 2, L, L) |> cu
    y = rand(N,2,L) |> cu
    return A,F1,F2,y
end

test_instance_tensor_long!(y, A, F1, F2) = @tullio y[xl, nl, l] = A[xi, ni, xj, nj, i, j] * F1[xl, nl, xj, nj, l, j]*F2[xi, ni, xl, nl, i, l]