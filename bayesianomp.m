function [ xhat ] = bayesianomp(...
    phi, y, stddev, stddevx, b, steps)
%bayesianomp Bayesian Orthogonal Matching Pursuit, according to
%  "Structured Bayesian Orthogonal Matching Pursuit", Dremeau et al.
%   Not as general as the implementation in the paper; it uses Bernoulli
%   priors over coefficient support (a special case of Bolzmann machines).
%
% Important: the assumption is that phi is unit l2-normalized accross columns
%
% Arguments:
% phi       ... dictionary, sensing matrix (IxM). Make sure the atoms have unit l2-norm.
% y         ... compressed representation (IxN), where N is the number of input signals
% stddev    ... standard deviation of noise
% stddevx   ... standard deviation of activations (given the coefficient is active)
% b         ... parameterization of Bernouli pdf, b = 1./(1+exp(-p)), where p is probability of `1`
% steps     ... number of iterations. Number of active atoms returned cannot exceed this value
%
% Returns:
% MxN sparse matrix of activation coefficients
%
% Note: convergence criterion is not implemented; based on stddev, stddevx
% and b, the algorithm may decide not to use additional atoms.
% Still, why not have it, based on the posterior probability - a TODO.

siglen = size(phi,2);
numsignals = size(y,2);

r = y;  % initial residual
shat = sparse(siglen, numsignals);
xhat = sparse(siglen, numsignals);
for step = 1:steps    
    
    %
    % Step 1. s macron
    %
    thresh = -2.*stddev.*(stddev./stddevx+1).*b;
    smacron = (phi.'*r + xhat).^2 > thresh;
    
    %
    % Step 2. Choose the atom to be modified
    %
    xmacron = smacron.*(phi.'*r+ xhat)./(stddev./stddevx+1);    
    argminof = 1./(2.*stddev).*(...
        sqrt(sum(r.^2) + 2.*phi.'*r.*(xhat-xmacron) + (xhat-xmacron).^2)) ...% We assume that sum(phi.^2) ==1 (third summand)
        - 1./(2.*stddevx).*(xhat.^2-xmacron.^2)....
        + (shat-smacron).*b;    
    [~, istar] = min(argminof);
    
    %
    % Step 3. Update the SR support
    %
    % Update shat to become smacron at indices where istar.
    idx = istar + (0:size(shat,1):(numel(shat)-1)); % a shortcut to get linear indices
    shat(idx) = smacron(idx);
    % TODO: use array indexing instead of linear indexing
    
    %
    % Step 4. Update the SR coefficients
    %
    % Here we have calculation of inverse, which will be done sequentially,
    % in a loop. Luckily, complexity of this step is a function of sparsity
    % and not a function of dimensionality of the problem, so it's not
    % prohibitively expensive.
    xhat = sparse(siglen, numsignals);
    for idx=1:numsignals
        support = logical(shat(:,idx));
        phisub = phi(:,support);
        xhat(support,idx) = (phisub.'*phisub + eye(size(phisub,2)).*stddev./stddevx)\phisub.'*y(:,idx);
        % xhat(support,idx) = inv(phisub.'*phisub + eye(size(phisub,2)).*stddev./stddevx)*phisub.'*y(:,idx);
    end
    
    %
    % Step 5: Update the residual
    %
    r = y - phi*xhat;
end

