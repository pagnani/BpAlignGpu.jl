function solmaxbel(af)
    @extract af : bpb
    @extract bpb : beliefs
    
    L = size(beliefs, 3)
    maxbel = Array{Float32}(undef, L)
    solbel = Array{CartesianIndex{2}}(undef, L)
    for i=1:L
        maxbel[i], solbel[i] = findmax(af.bpb.beliefs[:,:,i])
    end
    return solbel, maxbel 
end

function convert_soltosequence!(solbel, strseq, N, L)
    xnsol = fill((0,0), L)
    for i=1:L
        xnsol[i] = (solbel[i][2]-1, solbel[i][1]-1)
    end
    
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