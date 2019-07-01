%   %   %   %   %   %   %   %   %   %   %   %   %   %   %   %   %   %
% Generating input signals
%
siglen = 2048;
numsignals = 5;  % We'll process several signals, in paralell until we'll need inversed
targetsparsity = 64/siglen; % Expecting 64 active coefficient on average per signal
stddev = 0.01;
stddevx = 1;
%
% Sensing matrix
measurements = 256;
phi = randn(measurements, siglen);
normalizer = sqrt(sum(phi.^2));
phi = bsxfun(@rdivide, phi, normalizer);
%
% Activations/support, s
alpha = 0.1;    % using beta pdf to generate the Bernoulli priors.
p = betarnd(alpha.*ones(siglen, numsignals), alpha*(1-targetsparsity)./targetsparsity.*ones(siglen, numsignals));
s = sparse(rand(size(p))< p); % support
b = -log(1./p-1); % parameterization used by the model
%
% Magnitudes, x
x = sparse(siglen, numsignals);
x(s) = stddevx.*randn(nnz(s),1);
y = phi*x + stddev.*randn(size(phi,1), size(x,2));


%   %   %   %   %   %   %   %   %   %   %   %   %   %   %   %   %   %
% Bayesian OMP 
%
steps = 64 + 16;
recon = bayesianomp( phi, y, stddev, stddevx, b, steps)

signals_to_plot = [1,2,3];
for s = signals_to_plot
    subplot(numel(signals_to_plot), 1, s)
    original = stem(x(:,1), 'g.');
    hold on
    reconstructed = plot(recon(:,1), 'b.');
    hold off
    legend('oracle', 'reconstructed')
    title(strcat("Signal ", string(s)))
end
