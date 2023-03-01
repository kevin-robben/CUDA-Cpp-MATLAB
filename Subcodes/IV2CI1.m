function CI = IV2CI1(IV,dims)
    CI = 0;
    for i=1:numel(dims)
        dimProd = 1;
        for k=1:i-1
            dimProd = dimProd*dims(k);
        end
        CI = CI + (IV(i)-1)*dimProd;% -1 due to 1-based indexing
    end
    CI = CI + 1;% +1 due to 1-based indexing
end