function [buy, sell] = grid_arbi_rates(time, structure)
    if structure == 'A'
        buy = 0.15;
        sell = 0.15;
        return
    end
    time_ind = find([time < 8 time < 16 time < 20 1],1);
        
    if structure == 'B'
        buys = [0.1 0.15 0.3 0.1];
        sells = [0.1 0.15 0.3 0.1];
    else
        buys = [0.12 0.18 0.35 0.12];
        sells = [0.06 0.09 0.07 0.06];
    end
    buy = buys(time_ind);
    sell = sells(time_ind);
end