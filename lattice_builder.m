function node = lattice_builder(t,i)
    load("table.mat","table")
    node = table(i,t,:);
end