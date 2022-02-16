function SCE_update_conditional_lr!(conditional)

    L= size(conditional, 3)

    @inbounds for j=1:L-2
        for i=j+2:L
            C1 = view(conditional,:,:,i,i-1)
            C2 = view(conditional,:,:,i-1,j)
            conditional[:,:,i,j] .= C1*C2
        end
    end    
    @inbounds for j=3:L
        for i=j-2:-1:1
            C1 = view(conditional,:,:,i,i+1)
            C2 = view(conditional,:,:,i+1,j)
            conditional[:,:,i,j] .= C1*C2
        end
    end
end

