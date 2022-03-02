function solmaxbel(af)
    @extract af : bpb
    @extract bpb : beliefs
    
    L = size(beliefs, 3)
    maxbel = Array{Float32}(undef, L)
    solbel = Array{CartesianIndex{2}}(undef, L)
    xnsol = fill((0,0), L)
    for i=1:L
        maxbel[i], solbel[i] = findmax(af.bpb.beliefs[:,:,i])
        xnsol[i] = (solbel[i][2]-1, solbel[i][1]-1)
    end
    for i=1:L
    end
    return xnsol, maxbel 
end

function convert_soltosequence!(xnsol, strseq, N, L)
    
    pa = ""
    for i in 1:L
	    n = xnsol[i][2]
	    x = xnsol[i][1]
	    if x == 0
	        pa *= '-'
        else
	        pa *= strseq[n]
        end
    end
    po = ""
    i = 1
    start = false
    nold = 0
	f = 0
	l = 0
    while i <= L
        n = xnsol[i][2]
		x = xnsol[i][1]
        if x == 1 && start == true
            delta = n - nold - 1
            if delta > 0
                for d in 1:delta
                    po *= lowercase(strseq[nold+d])
                end
            end
            po *= strseq[n]
            nold = n
            l = n
        end
        if x == 1 && start == false
            start = true
            po *= strseq[n]
            nold = n
            f = n
        end
        if x == 0
            po *= "-"
        end
        i += 1
    end
    pa, po, f, l
end


function decodeposterior(P,strseq)

    N = length(strseq)
    L = length(P)
    pa = ""
    for i in 1:L
        maxP = -Inf
	    idxx = []
	    idxn = []
        for x in 0:1, n in 0:N+1
	        if P[i][x,n] > maxP
	            maxP = P[i][x,n]
	            idxx = x
	            idxn = n
	        end
        end
    	if maxP == -Inf
	       println("Problem with $i")
	    end
	    n = idxn
	    x = idxx
	    if x == 0
	        pa *= '-'
        else
	        pa *= strseq[n]
        end
    end
    po = ""
    i = 1
    start = false
    nold = 0
	f = 0
	l = 0
    while i <= L
        maxP = -Inf
	    idxx = []
	    idxn = []
	    for x in 0:1, n in 0:N+1
	        if P[i][x,n] > maxP
	            maxP = P[i][x,n]
	            idxx = x
	            idxn = n
	        end
	    end
        n = idxn
		x = idxx
        if x == 1 && start == true
            delta = n - nold - 1
            if delta > 0
                for d in 1:delta
                    po *= lowercase(strseq[nold+d])
                end
            end
            po *= strseq[n]
            nold = n
            l = n
        end
        if x == 1 && start == false
            start = true
            po *= strseq[n]
            nold = n
            f = n
        end
        if x == 0
            po *= "-"
        end
        i += 1
    end
    return pa, po ,f ,l
end
#--------------------------------------------------------------------#
function check_assignment(P,verbose,N)

    count = 0
    L = length(P)
    if verbose == true
	println("Let us check the assignment...")
    end
    for i in L:-1:2
        maxP = -Inf
	idxn = []
	idxx = []
	for x in 0:1, n in 0:N+1
	   if P[i][x,n] > maxP
	      maxP = P[i][x,n]
	      idxx = x
	      idxn = n
	   end
	end
	if maxP == -Inf
	   println("Problem with node $i")
	end
	n =idxn
	x =idxx
        maxP = -Inf
        idxn = []
        idxx = []
        for xj in 0:1, nj in 0:N+1
	    if P[i-1][xj,nj] > maxP
	       maxP = P[i-1][xj,nj]
	       idxx = xj
	       idxn = nj
	    end
	 end
	 nj = idxn
	 xj = idxx
	 sat = x == 1 ? (n > nj) : (n == nj || n == N+1)
        if sat == false && verbose == true
	   println("- $i → ($x, $n) $i-1 → ($xj, $nj)")
        end
        count += 1 - convert(Int, sat)
    end

    if count == 0
	if verbose == true
	    println("The subsequence satisfies the constraints")
	end
        return true
    else
	if verbose == true
	    println("There are $count short-range constraints not satisfied")
	end
	return false
    end

end

function check_sr!(xnsol, L, N)
    check = fill(0, L)
    #i=1
    if xnsol[1][1] == 0 && xnsol[1][2] != 0
        check[1]=1
    end
    if xnsol[1][1] == 1 && (xnsol[1][2] == 0 || xnsol[1][2] == N+1)
        check[1]=1
    end
    
    #i=2:L
    for i=2:L
        if xnsol[i-1][1]==0 && xnsol[i][1]==0
            if xnsol[i-1][2] != xnsol[i][2]
                check[i]=1
            end
        elseif xnsol[i-1][1]==1 && xnsol[i][1]==0
            if xnsol[i-1][2] != xnsol[i][2] && xnsol[i][2] != N+1
                check[i]=1
            end
        elseif xnsol[i-1][1]==0 && xnsol[i][1]==1
                if xnsol[i-1][2] < xnsol[i][2] && xnsol[i][2]<N+1
                else
                    check[i]=1
                end
        else
                if 0<xnsol[i-1][2] && xnsol[i-1][2] < xnsol[i][2] && xnsol[i][2]<N+1
                else
                    check[i]=1
                end
        end
    end
    
    #i=L
    if xnsol[L][1] == 0 && xnsol[L][2] != N+1
        check[L]+=1
    end
    if xnsol[L][1] == 1 && (xnsol[L][2] == 0 || xnsol[L][2] == N+1)
        check[L]+=1
    end
    check
end
#--------------------------------------------------------------------#
function compute_cost_function(J::Array{Float32,4}, h::Array{Float32,2}, seqins::String, L::Int, ctype::Symbol, λo::Vector{Float32}, λe::Vector{Float32}, μext::Float32, μint::Float32)

    seq = ""
    N = length(seqins)
    for i = 1:N
        if isuppercase(seqins[i]) || seqins[i] == '-'
	        seq *= seqins[i]
        end
    end
    #println(seqins)
    #println(seq)
    en = compute_potts_en(J,h,seq,L,ctype)
    inscost = 0.0 ### debug ###
    extend = false
    idx = 0
    first = true
    nopen = 0
    next = 0
    for i = 1:N
        if isuppercase(seqins[i]) || seqins[i] == '-'
	        idx += 1
	        extend = false
        elseif islowercase(seqins[i]) & extend == false
	        extend = true
	        nopen += 1
	        en += λo[idx+1]
            inscost += λo[idx+1] ### debug ###
	        #println("open ", idx+1, " ", λo[idx+1])
        elseif islowercase(seqins[i]) & extend == true
	        en += λe[idx+1]
            inscost += λe[idx+1] ### debug ###
	        next += 1
	        #println("ext ", idx+1, " ", λe[idx+1])
        end
    end
    ngaps_int = 0
    ngaps_ext = 0
    first = true
    for i = 1:L
        if seq[i] == '-'
	        if first
	            ngaps_ext += 1
	        else
	            ngaps_int += 1
	        end
        end
        if seq[i] != '-'
	        first = false
        end
    end
    first = true
    for i = L:-1:1
        if seq[i] == '-'
	        if first
	            ngaps_ext += 1
	            ngaps_int -= 1
	        end
        end
        if seq[i] != '-'
	        first = false
        end
    end
    #println(ngaps_ext, " gaps esterni e ", ngaps_int, " interni")
    #println(nopen, " open insertions plus ", next, " extended")
    gapcost = μext * ngaps_ext + μint * ngaps_int ### debug ###
    en += μext * ngaps_ext + μint * ngaps_int
    @show gapcost, inscost
    return en
end

function compute_potts_en(J::Array{Float32,4}, h::Array{Float32,2}, seq::String, L::Int,ctype::Symbol)

    en = 0
    onesite = 0.0 ### debug ###
    couplings = 0.0 ### debug ###
    for i in 1:L
        Aᵢ = letter2num(seq[i],ctype)
        en += -h[Aᵢ,i]
        onesite += -h[Aᵢ,i]
        for j in i+1:L
            Aⱼ = letter2num(seq[j],ctype)
            en += -J[Aᵢ,Aⱼ,i,j]
            couplings += -J[Aᵢ,Aⱼ,i,j]
        end
    end
    @show onesite, couplings
    return en
end
