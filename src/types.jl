struct Seq
    header::String
    strseq::String
    intseq::Vector{Int}
    ctype::Symbol
    function Seq(h::String, s::String, v::Vector{Int}, ctype::Symbol)
        if length(s) != length(v)
            error("length s != length v")
        end
        for i in eachindex(s) #check alignment between strseq and inteseq
            if letter2num(s[i], ctype) != v[i]
                error("intseq not representing strseq at index $i")
            end
        end
        return new(h::String, s::String, v::Vector{Int}, ctype::Symbol)
    end
end

function Seq(h::String, s::String, ctype::Symbol)
    v = fill(-1, length(s))
    s = Vector{Char}(s)
    if ctype == :nbase
        replace!(s, 'T' => 'U')
    end
    for i in eachindex(s)
        v[i] = letter2num(s[i], ctype)
    end
    s = String(s)
    Seq(h, s, v, ctype)
end

function Base.show(io::IO, x::Seq)
    println(io, x.header)
    #print_with_color(:cyan,io,x.strseq)
    println(io, x.strseq)
end

struct ParamAlgo
    damp::Float64
    tol::Float64
    tolnorm::Float64
    tmax::Int
    upscheme::Symbol
    lr::Symbol
    beta::Float64
    verbose::Bool
end

struct ParamModel{T}
    N::Int
    L::Int
    q::Int
    muint::T
    muext::T
    lambda_o::Vector{T}
    lambda_e::Vector{T}
    H::Matrix{T}
    J::Array{T,4}
end

struct BPMessages{T2,T3,T6}
    F::T3
    B::T3
    hF::T3
    hB::T3
    scra::T2
    Hseq::T3
    Jseq::T6
end

function BPMessages(seq::Seq, para::ParamModel; T = Float32, ongpu=true)
    @extract seq:intseq
    @extract para:J H q
    L = size(H, 2)
    N = length(intseq)
    gpufun = ongpu ? cu : identity
    Hseq = zeros(T, N + 2, 2, L)
    Jseq = zeros(T, N + 2, 2, N + 2, 2, L, L)

    for i in 1:L
        for xi in 1:2
            for ni in 1:N+2
                if xi == 1
                    Hseq[ni, xi, i] = T(H[q, i])
                else
                    if ni < 2 || ni > N + 1
                        Hseq[ni, xi, i] = T(0)
                    else
                        Hseq[ni, xi, i] = T(H[intseq[ni-1], i])
                    end
                end
            end
        end
    end
    # case xi == xj == 1
    for i in 1:L
        for j in 1:L
            for ni in 1:N+2
                for nj in 1:N+2
                    Jseq[nj, 1, ni, 1, j, i] = T(J[q, q, j, i])
                end
            end
        end
    end

    # case xj == 1 xi == 2 
    for i in 1:L
        for j in 1:L
            for ni in 2:N+1
                for nj in 1:N+2
                    Jseq[nj, 1, ni, 2, j, i] = T(J[q, intseq[ni-1], j, i])
                end
            end
        end
    end
    # case xj == 2 xi == 1
    for i in 1:L
        for j in 1:L
            for ni in 1:N+2
                for nj in 2:N+1
                    Jseq[nj, 2, ni, 1, j, i] = T(J[intseq[nj-1], q, j, i])
                end
            end
        end
    end

    # case xj == 2 xi == 2
    @time for i in 1:L
        for j in 1:L
            for ni in 2:N+1
                for nj in 2:N+1
                    Jseq[nj, 2, ni, 2, j, i] = T(J[intseq[nj-1], intseq[ni-1], j, i])
                end
            end
        end
    end

    F = rand(T, N + 2, 2, L) # forward message, from variable to factor
    B = rand(T, N + 2, 2, L)  # backward message, from variable to factor
    hF = rand(T, N + 2, 2, L) # forward message, from factor to variable
    hB = rand(T, N + 2, 2, L) # backward message, from factor to variable
    scra = zeros(T, N + 2, 2) #used in updates for BP messages

    for v in (F, B, hF, hB)
        v[1, 2, :] .= T(0)
        v[N+2, 2, :] .= T(0)
    end

    for v in (F, B, hF, hB)
        mynorm = sum(v, dims = (1, 2))[:]
        for i in 1:L
            for xi in 1:2
                for ni in 1:N+2
                    v[ni, xi, i] /= mynorm[i]
                end
            end
        end
    end

    rF = F |> gpufun
    rB = B |> gpufun
    rhB = hB |> gpufun
    rhF = hF |> gpufun
    rscra = scra |> gpufun
    rJseq = Jseq |> gpufun
    rHseq = Hseq |> gpufun

    T2 = typeof(rscra)
    T3 = typeof(rF)
    T6 = typeof(rJseq)
    return BPMessages{T2,T3,T6}(rF, rB, rhF, rhB, rscra, rHseq, rJseq)
end

function Base.show(io::IO, x::BPMessages)
    n, _, L = size(x.F)
    N = n - 2
    isgpu = typeof(x.F) <: CuArray
    println(io, "BPMessages{$(eltype(x.F))}[L=$L N=$N ongpu=$isgpu]")
end

struct BPBeliefs{T3,T5,T6}
    beliefs::T3
    beliefs_old::T3
    joint_chain::T5
    conditional::T6
end

function BPBeliefs(N::Int, L::Int; gpu::Bool = true, T::DataType = Float32)
    gpufun = gpu ? cu : identity

    beliefs = rand(T, N + 2, 2, L)
    beliefs[1, 2, :] .= T(0)
    beliefs[N+2, 2, :] .= T(0)
    beliefs[2:N+2, 1, 1] .= T(0)
    #mynorm = sum(beliefs, dims = (1, 2))[:]
    beliefs_old = rand(T, N + 2, 2, L)
    joint_chain = zeros(T, N + 2, 2, N + 2, 2, L)
    conditional = zeros(T, N + 2, 2, N + 2, 2, L, L)


    mynorm = sum(beliefs, dims = (1, 2))[:]
    for i in 1:L
        for xi in 1:2
            for ni in 1:N+2
                beliefs[ni, xi, i] /= mynorm[i]
            end
        end
    end

    # for i in 1:L
    #     for j in 1:L
    #         for xi in 1:2
    #             for ni in 1:N+2
    #                 for xj in 1:2
    #                     for nj in 1:N+2
    #                         if i == j
    #                             if xi == xj && ni == nj
    #                                 conditional[nj, xj, ni, xi, j, i] = T(1)
    #                             else
    #                                 conditional[nj, xj, ni, xi, j, i] = T(0)
    #                             end
    #                         end
    #                     end
    #                 end
    #             end
    #         end
    #     end
    # end

    T3 = typeof(beliefs |> gpufun)
    T5 = typeof(joint_chain |> gpufun)
    T6 = typeof(conditional |> gpufun)

    BPBeliefs{T3,T5,T6}(beliefs |> gpufun, beliefs_old |> gpufun, joint_chain |> gpufun, conditional |> gpufun)
end

struct LongRangeFields{T3,T5}
    f::T3
    g::T5
end

function LongRangeFields(N::Integer, L::Integer; gpu::Bool = true, T::DataType = Float32) 
    gpufun = gpu ? cu : identity

    f = zeros(T, N + 2, 2, L)
    g = zeros(T, N + 2, 2, N + 2, 2, L)

    rf = f |> gpufun
    rg = g |> gpufun
    RT3 = typeof(rf)
    RT5 = typeof(rg)
    LongRangeFields{RT3,RT5}(rf, rg)
end

struct allFields{T2,T3,T5,T6}
    bpm::BPMessages{T2,T3}
    bel::BPBeliefs{T3,T5,T6}
    lrf::LongRangeFields{T3,T5}
end