% @Author: hexy_
% @Date:   2020-12-05 11:08:50
% @Last Modified by:   Xiaoyu He
% @Last Modified time: 2021-03-23 11:30:45

% evolutionary gradient search with limited memory matrix adaptation
function [x, infos, opts] = SEGS_LMMA(problem, in_opts)
% set dimensions and samples
n = problem.dim();
N = problem.samples();  
lambda = 4+floor(3*log(n));

% set local options 
local_opts.batch_size = 50;
local_opts.stepsize_eta = 1e0;
local_opts.stepsize_b0 = 1e0;
local_opts.mu = 1e-8;

% merge options
opts = mergeOptions(ESO_default_opts(problem), local_opts);   
opts = mergeOptions(opts, in_opts);  
T = opts.maxFEs / opts.batch_size / (lambda+1);
mu = opts.mu;
iter = 0; FEs = 0;
stepsize_bn = opts.stepsize_b0;
assert(opts.parallel_eval==false);

m = lambda;
cd = 1/n./1.5.^(0:m-1);
cc = lambda/n./4.^(0:m-1);
M = zeros(n,m);

x = opts.x0; 
ep = ones(n,1) * 1e-8;
ep_len = norm(ep);

algName = 'SEGS-LMMA';

[infos, f_val] = record_info(problem, x, opts, [], FEs, 0);        
if opts.verbose > 0
    fprintf('%s: f0 = %g, mu0 = %g, eta = %g, b0 = %g\n', algName, f_val, mu, opts.stepsize_eta, stepsize_bn);
end
% main loop
elapsed_time = 0;
logModule = ceil(N/(lambda+1)/opts.batch_size/opts.logPerEpoch);
while FEs < opts.maxFEs
    start_time = tic();

    % sub-sampling
    idxs = randperm(N,opts.batch_size);

    % estimate natural gradient
    arZ = randn(n,lambda);
    arU = arZ;
    for j = 1 : min(m,iter)
        arU = (1-cd(j))*arU+cd(j)*M(:,j).*(M(:,j)'*arU);
    end

    % forward gradient estimation
    fx = problem.cost_batch(x,idxs);
    f_trial = zeros(lambda,1);
    for i = 1 : lambda
        f_trial(i) = problem.cost_batch(x + arU(:,i) * mu,idxs);
    end
    grad = arU * (f_trial - fx) / mu / lambda;
    FEs = FEs + opts.batch_size * (lambda+1);

    % mutation step that increases the likelihood of reproducing better solutions
    %   it is exactly the natural gradient of the cholesky factor of the covariance at the point of I
    [~, arIndex] = sort(f_trial); arIndex = arIndex(1:floor(lambda/2));
    mutant_step = sqrt(length(arIndex)) * mean(arZ(:,arIndex),2);

    % step size adaptation: 
    grad_r = grad;
    avg_v2 = 0;
    for j = min(m,iter) : -1 : 1
        v = M(:,j);
        v2 = v'*v;
        grad_r = 1/(1-cd(j))*(grad_r - cd(j)/(1-cd(j)+cd(j)*v2)*v*(v'*grad_r));
        avg_v2 = avg_v2 + v2/min(m,iter);
    end
    stepsize_bn = sqrt(stepsize_bn^2 + grad_r'*grad_r);

    sigma = opts.stepsize_eta/stepsize_bn;

    % descent
    x = x - grad * sigma;

    % metric learning
    M = (1-cc).*M + sqrt(cc.*(2-cc)).*mutant_step;

    % measure elapsed time
    elapsed_time = elapsed_time + toc(start_time);

    iter = iter + 1;
    epochs = FEs/N;
    if mod(iter,logModule) == 1
        % store infos
        [infos, f_val] = record_info(problem, x, opts, infos, FEs, elapsed_time);        
        if opts.verbose > 0
            fprintf('%s: epochs = %.1f, fobj = %g, step = %g, |g| = %g, |v| = %g\n', algName, epochs, f_val, sigma, norm(grad), sqrt(avg_v2));
        end
    end
end