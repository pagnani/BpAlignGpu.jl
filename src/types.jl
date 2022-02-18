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

const allowed_upschemes = [:random,:sequential]
const allowed_lrs = [:sce,:mf]
mutable struct ParamAlgo{T<:AbstractFloat}
    damp::T
    tol::T
    tolnorm::T
    tmax::Int
    upscheme::Symbol # :random or :sequential
    lr::Symbol  # :sce or :mf 
    beta::T
    verbose::Bool
    function ParamAlgo(damp::T, tol::T,tolnorm::T, tmax::Int, upscheme::Symbol, lr::Symbol, beta::T, verbose::Bool) where {T<:AbstractFloat}
        upscheme in allowed_upschemes || error("only upscheme ∈ $allowed_upschemes allowed")
        lr in allowed_lrs || error("only lr ∈ $allowed_lrs allowed")
        return new{T}(damp, tol,tolnorm, tmax, upscheme, lr, beta, verbose)
    end
end

function Base.show(io::IO, x::ParamAlgo{T}) where {T<:AbstractFloat}
    println(typeof(x))
    println("-------------")
    for i in fieldnames(ParamAlgo)
        println(i, "=", getfield(x, i))
    end
    println("-------------")
end

function ParamAlgo{RT}(x::ParamAlgo{T}) where {T<:AbstractFloat, RT<:AbstractFloat}  
    RT === T && return x
    return ParamAlgo(RT(x.damp), RT(x.tol),RT(x.tolnorm), x.tmax, x.upscheme, x.lr, RT(x.beta), x.verbose)
end

function Base.convert(::Type{RT},x::ParamAlgo{T}) where {RT <: ParamAlgo, T<:AbstractFloat}
    x isa RT && return x
    return RT(x)
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

function ParamModel{RT}(x::ParamModel{T}) where {RT <: AbstractFloat,T <: AbstractFloat}
    RT === T && return x
    return ParamModel(x.N, x.L, x.q, RT(x.muint), RT(x.muext), Vector{RT}(x.lambda_o), Vector{RT}(x.lambda_e), Matrix{RT}(x.H), Array{RT,4}(x.J))
end

function Base.convert(::Type{RT},x::ParamModel{T}) where {RT<:ParamModel,T<:AbstractFloat}
    x isa RT && return x
    return RT(x) 
end

function Base.show(io::IO, x::ParamModel{T}) where {T}
    q, L = size(x.H)
    println(io, "ParamModel{$(eltype(x.H))}[L=$(x.L) N=$(x.N) q=$(x.q)]")
end

struct BPMessages{T1,T3,T6}
    F::T3
    B::T3
    hF::T3
    hB::T3
    scra::T3
    Hseq::T3
    Jseq::T6
    lambda_e::T1
    lambda_o::T1
end

function BPMessages(seq::Seq, para::ParamModel; T = Float32, ongpu = true)
    @extract seq : intseq
    @extract para : J H q lambda_o lambda_e
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
    for i in 1:L
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
    scra = zeros(T, N + 2, 2, L) #used in updates for BP messages

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
    rlambda_e = lambda_e |> gpufun
    rlambda_o = lambda_o |> gpufun

    T1 = typeof(rlambda_e)    
    T3 = typeof(rF)
    T6 = typeof(rJseq)
    return BPMessages{T1,T3,T6}(rF, rB, rhF, rhB, rscra, rHseq, rJseq,rlambda_e,rlambda_o)
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

    for i in 1:L
        for xi in 1:2
            for ni in 1:N+2
                conditional[ni, xi, ni, xi, i, i] = T(1)
            end
        end
    end

    rbeliefs = beliefs |> gpufun
    rbeliefs_old = beliefs_old |> gpufun
    rjoint_chain = joint_chain |> gpufun
    rconditional = conditional |> gpufun
    T3 = typeof(rbeliefs)
    T5 = typeof(rjoint_chain)
    T6 = typeof(rconditional)

    return BPBeliefs{T3,T5,T6}(rbeliefs, rbeliefs_old, joint_chain, rconditional)
end

function Base.show(io::IO, x::BPBeliefs)
    isgpu = typeof(x.beliefs) <: CuArray
    n, _, L = size(x.beliefs)
    N = n - 2
    print(io, "BPBeliefs{$(eltype(x.beliefs))}[L=$L N=$N ongpu=$isgpu]")
end

struct LongRangeFields{T3,T5}
    f::T3
    g::T5
end

function LongRangeFields(N::Integer, L::Integer; ongpu::Bool = true, T::DataType = Float32)
    gpufun = ongpu ? cu : identity

    f = zeros(T, N + 2, 2, L)
    g = zeros(T, N + 2, 2, N + 2, 2, L)

    rf = f |> gpufun
    rg = g |> gpufun
    RT3 = typeof(rf)
    RT5 = typeof(rg)
    LongRangeFields{RT3,RT5}(rf, rg)
end

function Base.show(io::IO, x::LongRangeFields)
    isgpu = typeof(x.f) <: CuArray
    n, _, L = size(x.f)
    N = n - 2
    print(io, "LongRangeFields{$(eltype(x.f))}[L=$L N=$N ongpu=$isgpu]")
end
struct AllFields{T1,T2,T3}
    bpm::T1
    bpb::T2
    lrf::T3
    function AllFields(bpm::T1, bpb::T2, lrf::T3) where {T1<:BPMessages,T2<:BPBeliefs,T3<:LongRangeFields}
        isgpu(x) = typeof(x) <: CuArray
        isgpubpm = isgpu(bpm.B)
        isgpubpb = isgpu(bpb.beliefs)
        isgpulrf = isgpu(lrf.f)
        isgpubpm == isgpubpb == isgpulrf || error("all fields should be either on gpu or on cpu")
        return new{T1,T2,T3}(bpm,bpb,lrf)
    end
end

function Base.show(io::IO, x::AllFields)
    isgpu = typeof(x.bpm.B) <: CuArray
    n, _, L = size(x.lrf.f)
    N = n-2
    print(io, "AllFields{$(eltype(x.bpm.B))}[L=$L N=$N ongpu=$isgpu]")
end