%   %   %   %   %   %   %   %   %   %   %   %   %   %   %   %   %   %
% Generating input signals
%
siglen = 2048;
numsignals = 3;  % We'll process several signals (in paralell, until to point we'll need inverses/projections)
targetsparsity = 64/siglen; % Expecting 64 active coefficient on average per signal
stddevsq = 0.1; % Relatively hgh noise
stddevsqx = 1;
%
% Sensing matrix
measurements = 256;
phi = randn(measurements, siglen);
normalizer = sqrt(sum(phi.^2));
phi = bsxfun(@rdivide, phi, normalizer);
%
% Activations/support, s
alpha = 0.01;    % using beta pdf to generate the Bernoulli priors.
p = betarnd(alpha.*ones(siglen, numsignals), alpha*(1-targetsparsity)./targetsparsity.*ones(siglen, numsignals));
s = sparse(rand(size(p))< p); % support
b = -log(1./p-1); % parameterization used by the model
%
% Magnitudes, x
x = sparse(siglen, numsignals);
x(s) = stddevsqx.*randn(nnz(s),1);
y = phi*x + stddevsq.*randn(size(phi,1), size(x,2));


%   %   %   %   %   %   %   %   %   %   %   %   %   %   %   %   %   %
% Bayesian OMP 
%
steps = 64 + 16;
recon = bayesianomp( phi, y, stddevsq, stddevsqx, b, steps, 64);
recon_no_prior = bayesianomp( phi, y, stddevsq, stddevsqx, b*0, steps, 64);

signals_to_plot = [1,2,3];
for s = signals_to_plot
    % No prior used
    subplot(numel(signals_to_plot), 2, s*2-1)
    original = stem(x(:,s), 'g.');
    hold on
    reconstructed = plot(recon_no_prior(:,s), 'b.');
    hold off
    legend('oracle', 'reconstructed')
    title(strcat("Activations of signal ", string(s), ", no prior"))
end
%
for s = signals_to_plot
    % With prior knowleedge of activations
    subplot(numel(signals_to_plot), 2, s*2)
    original = stem(x(:,s), 'g.');
    hold on
    reconstructed = plot(recon(:,s), 'b.');
    hold off
    legend('oracle', 'reconstructed')
    title(strcat("Activations of signal ", string(s), ", with prior"))
end