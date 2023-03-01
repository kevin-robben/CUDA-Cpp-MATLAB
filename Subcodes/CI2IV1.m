function IV = CI2IV1(CI,dims)
    IV = zeros(numel(dims),1);
    for i=1:numel(dims)
        dimProd = 1;
        for k=1:i-1
            dimProd = dimProd*dims(k);
        end
        IV(i) = mod(floor((CI-1)/dimProd),dims(i))+1;% -1 and +1 due to 1-based indexing
    end
end