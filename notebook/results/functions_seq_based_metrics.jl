function hamming_distance(s1::String, s2::String)
    l1 = length(s1)
    l2 = length(s2)
    if l1 != l2
        println("seqs must have same length")
    end
    n = 0
    for i in 1:l1
         if s1[i] != s2[i]
            n += 1
        end
    end
    return n
end

function count_gap_pm(sref::String, star::String)
    l1 = length(sref)
    l2 = length(star)
    if l1 != l2
        println("seqs must have same length")
    end
    np = 0
    nm = 0
    for i in 1:l1
        if sref[i] !== '-' && star[i] == '-'
            np += 1
        end
        if sref[i] == '-' && star[i] !== '-'
            nm += 1
        end
    end
    return np, nm
end

function count_mismatch(s1::String, s2::String)
    l1 = length(s1)
    l2 = length(s2)
    if l1 != l2
        println("seqs must have same length")
    end
    m = 0
    for i in 1:l1
        if s1[i] !== '-' && s2[i] !== '-'
            #@show i, s1[i], s2[i] 
            if s1[i] !== s2[i]
                m += 1
            end
        end
    end
    return m
end