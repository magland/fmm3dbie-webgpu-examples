% Point source in an ellipsoidal room with absorbing walls (interior problem).

mip install --channel mip-org/labs fmm3dbie-webgpu
mip load fmm3dbie-webgpu

abc = [2; 1.5; 1];   % room semi-axes
zk = 4;              % wavenumber
beta = 0.5;          % wall admittance: du/dn = 1i*zk*beta*u on the walls
lam = -beta;         % fmm3dbie's form du/dn + 1i*zk*lam*u = f
S = quad_ellipsoid(abc, 3, 6);
fprintf('room: %d patches, %d points\n', S.npatches, S.npts);
opts = struct('ifout', 0);   % interior problem

G = @(x, x0) exp(1i*zk*vecnorm(x - x0))./(4*pi*vecnorm(x - x0));
dGdn = @(x, n, x0) sum((x - x0).*n, 1).*(1i*zk*vecnorm(x - x0) - 1) ...
  .*exp(1i*zk*vecnorm(x - x0))./(4*pi*vecnorm(x - x0).^3);

% 1. Self-test: the field of a source outside the room solves the interior
%    problem with its own wall data.
xo = [3; 2; 1];
rhs = dGdn(S.r, S.n, xo) + 1i*zk*lam*G(S.r, xo);
t = tic;
[dens, errs, ~, Q] = helm3d.impedance.solver(S, lam, rhs, 1e-6, zk, 1, opts);
fprintf('self-test solve: %d GMRES iterations, %.1f s\n', numel(errs), toc(t));
chk = [0 0 0; 1 0.5 0.2; -1.2 -0.6 0.3; 0.4 -1 -0.5].';
p = helm3d.impedance.eval(S, dens, chk, 1e-6, zk, 1, struct('precomp_quadrature', Q));
fprintf('self-test: max relative error %.1e\n', max(abs(p.' - G(chk, xo)))/max(abs(G(chk, xo))));

% 2. Source in the room: u = G(x, x0) + v, with v from the impedance problem
x0 = [-0.83; 0.41; 0];
rhs = -(dGdn(S.r, S.n, x0) + 1i*zk*lam*G(S.r, x0));
opts.quadrature_correction = Q;
t = tic;
[dens, errs] = helm3d.impedance.solver(S, lam, rhs, 1e-6, zk, 1, opts);
fprintf('room solve: %d GMRES iterations, %.1f s\n', numel(errs), toc(t));

n = 101;
[X, Y] = meshgrid(linspace(-abc(1), abc(1), n), linspace(-abc(2), abc(2), n));
targ = [X(:).'; Y(:).'; zeros(1, n^2)];
in = (X(:).'/abc(1)).^2 + (Y(:).'/abc(2)).^2 < 0.95^2;
t = tic;
v = zeros(n^2, 1);
v(in) = helm3d.impedance.eval(S, dens, targ(:, in), 1e-6, zk, 1, struct('precomp_quadrature', Q));
fprintf('eval at %d points: %.1f s\n', nnz(in), toc(t));
fmm3dbie_gpu('free', Q.gpu_handle);

% Clip the real part near the source, where it is singular
u = max(min(real(G(targ, x0).' + v), 0.06), -0.06);
u(~in) = NaN;
figure;
imagesc(X(1, :), Y(:, 1), reshape(u, n, n));
axis xy equal tight;
colorbar;
hold on;
s = linspace(0, 2*pi, 200);
plot(abc(1)*cos(s), abc(2)*sin(s), 'k', 'LineWidth', 1.5);
plot(x0(1), x0(2), 'k*');
hold off;
title(sprintf('Re u at z = 0, room with absorbing walls, k = %g', zk));
xlabel('x');
ylabel('y');

% Ellipsoid with semi-axes abc as a surfer of quad patches.
function S = quad_ellipsoid(abc, na, norder)
  S0 = quad_sphere(1, na, [0; 0; 0], norder);
  D = diag(abc);
  r = D*S0.r;
  du = D*S0.du;
  dv = D*S0.dv;
  nrm = cross(du, dv);
  nrm = nrm./vecnorm(nrm);
  S = surfer(S0.npatches, norder, [r; du; dv; nrm], 11);
end
