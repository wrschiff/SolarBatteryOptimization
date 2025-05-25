clc ; close all ; clear all ;

H = 24;

% Creating a simple 24 stages lattice with 25 nodes at second stage
lattice = Lattice.latticeEasy(H, 25, @lattice_builder) ;

% Visualisation
figure ;
lattice.plotLattice(@(data) num2str(data)) ;

% Run SDDP
params = sddpSettings('algo.McCount',25, ...
                      'stop.iterationMax',10,...                      
                      'stop.pereiraCoef',2,...                   
                      'solver','gurobi') ;
var.pcd = sddpVar(H);
var.energy = sddpVar(H);
lattice = compileLattice(lattice,@(scenario)nlds(scenario,var,'A'),params) ;                                    
output = sddp(lattice,params) ;

% Visualise output
plotOutput(output) ;

% Forward passes
% lattice = output.lattice ;
% nForward = 5 ;
% objVec = zeros(nForward,1);
% x = zeros(nForward,H);
% y = zeros(nForward,H);
% p = zeros(nForward,H);
% for  i = 1:nForward
%     [objVec(i),~,~,solution] = forwardPass(lattice,'random',params) ;    
%     x(i,:) = lattice.getPrimalSolution(var.x, solution) ;
%     y(i,:) = lattice.getPrimalSolution(var.y, solution) ;
%     p(i,:) = lattice.getPrimalSolution(var.p, solution) ;
% end
