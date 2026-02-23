function beta = betaKMM(Xtrn, Ytrn, Xtst, Ytst, sigma, B, eps)
% betaKMM  Kernel Mean Matching weights (Gretton et al. style)
%
% This returns weights beta for the *training/source* sample Xtrn so that
% the reweighted source feature mean matches the target feature mean Xtst
% in RKHS with an RBF kernel.
%
% Signature matches your call:
%   beta = betaKMM(Xtrn, Ytrn, Xtst, Ytst, sigma, 1, 1);
% Ytrn and Ytst are not used (KMM is unsupervised w.r.t. labels).
%
% Requires Optimization Toolbox (quadprog).

    %#ok<*INUSD>  % silence unused variable warnings for Ytrn, Ytst

    if nargin < 6 || isempty(B),   B = 1000; end
    if nargin < 7 || isempty(eps), eps = 0.1; end

    Xtrn = double(Xtrn);
    Xtst = double(Xtst);

    ns = size(Xtrn,1);
    nt = size(Xtst,1);

    if ns == 0
        error('Xtrn is empty. No selected training observations.');
    end
    if nt == 0
        error('Xtst is empty. No selected test/target observations.');
    end

    % --- RBF kernel helper ---
    % K(x,z) = exp(-||x-z||^2 / (2*sigma^2))
    if sigma <= 0
        error('sigma must be > 0');
    end

    % Compute K_ss (ns x ns)
    % Efficient squared distance:
    Xs2 = sum(Xtrn.^2, 2);
    Dss = Xs2 + Xs2' - 2*(Xtrn*Xtrn');
    Kss = exp(-Dss./(2*sigma^2));

    % Compute kappa = (ns x 1) where kappa_i = (ns/nt) * sum_j K(x_i, x'_j)
    Xt2 = sum(Xtst.^2, 2);
    Dst = Xs2 + Xt2' - 2*(Xtrn*Xtst');
    Kst = exp(-Dst./(2*sigma^2));
    kappa = (ns/nt) * sum(Kst, 2);

    % Quadratic program:
    % minimize (1/2) beta' Kss beta - kappa' beta
    H = (Kss + Kss')/2;                 % make symmetric
    f = -kappa;

    % Constraints:
    % 0 <= beta_i <= B
    lb = zeros(ns,1);
    ub = B * ones(ns,1);

    % |sum(beta) - ns| <= ns*eps  -> two inequalities
    A = [ ones(1,ns); -ones(1,ns) ];
    b = [ ns*(1+eps); -ns*(1-eps) ];

    opts = optimoptions('quadprog','Display','off');

    % Add tiny ridge for numerical stability (optional)
    H = H + 1e-10*eye(ns);

    beta = quadprog(H, f, A, b, [], [], lb, ub, [], opts);

    if isempty(beta)
        error('quadprog failed to find a solution. Try increasing B or eps, or check Optimization Toolbox.');
    end

    % Ensure column vector
    beta = beta(:);
end