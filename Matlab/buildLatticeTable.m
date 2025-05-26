city = "Seattle";
nLoads = 5;
nIrrads = 5;

if city == "Phoenix"
    minVars = [0.8 0.9 0.85 0.8];
    maxVars = [1.2 1.1 1.15 1.2];
    means = [0 0 0 0 0.032 0.178 0.410 0.632 ...
        0.812 0.942 1.016 1.028 0.974 0.862 0.698 0.498 0.285 0.124 ...
        0.018 0 0 0 0];
elseif city == "Sacramento"
    minVars = [0.7 0.8 0.75 0.7];
    maxVars = [1.3 1.2 1.25 1.3];
    means = [0 0 0 0 0 0.015 0.142 0.356 0.556 ...
        0.712 0.825 0.891 0.902 0.855 0.756 0.612 0.436 0.245 0.098 ...
        0.008 0 0 0 0];
else
    minVars = [0.6 0.65 0.6 0.55];
    maxVars = [1.40 1.35 1.40 1.45];
    means = [0 0 0 0 0 0 0.072 0.224 0.367 0.498 ...
        0.594 0.654 0.676 0.644 0.562 0.442 0.302 0.158 0.054 0 ...
        0 0 0 0];
end
consump = [0.52 0.42 0.38 0.35 0.32 0.38 0.62 0.98 0.85 0.68 0.62 0.65 ...
    0.75 0.68 0.65 0.72 0.95 1.42 1.95 1.65 1.38 1.15 0.88 0.65];
consumpVarMin = 0.8;
consumpVarMax = 1.2;

table = zeros([nLoads*nIrrads,24,2]);
for hr=1:24
    cons = linspace(consumpVarMin*consump(hr),...
        consumpVarMax*consump(hr),nLoads);
    if hr < 8
        varStage = 1;
    elseif hr < 12
        varStage = 2;
    elseif hr < 16
        varStage = 3;
    else
        varStage = 4;
    end
    lv = minVars(varStage);
    hv = maxVars(varStage);
    irrs = linspace(lv*means(hr),...
        hv*means(hr),nIrrads);
    table(:,hr,:) = table2array(combinations(cons,irrs));
end
save('table.mat','table')
