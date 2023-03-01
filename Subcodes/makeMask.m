function mask = makeMask(N,US)
%%initialize mask indices vector
mask_indices = zeros(round(N/US),1);
%% for-loop over every 1-element that we'll choose in our mask
for i=1:size(mask_indices,1)
    bad_choice = 1;
    while(bad_choice == 1)
        mask_indices(i) = round(N/6*abs(randn));%
        if(i<=10)
            mask_indices(i) = i;
        end
        bad_choice = 0;
        for k=1:i
            if(k<i)
                if(mask_indices(i) == mask_indices(k))
                    bad_choice = 1;
                end
            end
            if(mask_indices(i) > N)
                bad_choice = 1;
            end
            if(mask_indices(i) == 0)
                bad_choice = 1;
            end
        end
    end
end
mask = zeros(N,1);
for i=1:size(mask_indices,1)
    mask(mask_indices(i)) = 1;
end

end