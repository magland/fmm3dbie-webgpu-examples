% Plane wave on a sound-hard ellipsoid, after a point-source self-test.

mip install --channel mip-org/labs fmm3dbie-webgpu
mip load fmm3dbie-webgpu

abc = [1.5; 0.8; 0.6];   % semi-axes
zk = 5;                  % wavenumber
S = quad_ellipsoid(abc, 3, 6);
fprintf('ellipsoid: %d patches, %d points\n', S.npatches, S.npts);

% 1. Self-test with a point source inside
x0 = [0.4; -0.2; 0.1];
G = @(x) exp(1i*zk*vecnorm(x - x0))./(4*pi*vecnorm(x - x0));
dGdn = @(x, n) sum((x - x0).*n, 1).*(1i*zk*vecnorm(x - x0) - 1) ...
  .*exp(1i*zk*vecnorm(x - x0))./(4*pi*vecnorm(x - x0).^3);
t = tic;
[dens, errs, ~, Q] = helm3d.neumann.solver(S, dGdn(S.r, S.n), 1e-6, zk, 1);
fprintf('self-test solve: %d GMRES iterations, %.1f s (includes near-quadrature setup)\n', ...
  numel(errs), toc(t));
chk = [3 0 0; 0 2 1; -1 -1.5 -1; 0.5 0.3 1.2].';
p = helm3d.neumann.eval(S, dens, chk, 1e-6, zk, 1, struct('precomp_quadrature', Q));
fprintf('self-test: max relative error %.1e\n', max(abs(p.' - G(chk)))/max(abs(G(chk))));

% 2. Plane wave, oblique incidence
d = [1; 0.6; 0]/norm([1; 0.6; 0]);
uinc = @(x) exp(1i*zk*(d.'*x));
rhs = -1i*zk*(d.'*S.n).*uinc(S.r);
t = tic;
[dens, errs] = helm3d.neumann.solver(S, rhs, 1e-6, zk, 1, struct('quadrature_correction', Q));
fprintf('plane wave solve: %d GMRES iterations, %.1f s (near quadrature reused)\n', ...
  numel(errs), toc(t));

n = 121;
L = 3;
[X, Y] = meshgrid(linspace(-L, L, n));
targ = [X(:).'; Y(:).'; zeros(1, n^2)];
out = (X(:).'/abc(1)).^2 + (Y(:).'/abc(2)).^2 > 1.02^2;
t = tic;
us = zeros(n^2, 1);
us(out) = helm3d.neumann.eval(S, dens, targ(:, out), 1e-6, zk, 1, struct('precomp_quadrature', Q));
fprintf('eval at %d points: %.1f s\n', nnz(out), toc(t));
fmm3dbie_gpu('free', Q.gpu_handle);

u = uinc(targ).' + us;
u(~out) = NaN;
figure;
imagesc(linspace(-L, L, n), linspace(-L, L, n), reshape(real(u), n, n));
axis xy equal tight;
colorbar;
hold on;
s = linspace(0, 2*pi, 200);
plot(abc(1)*cos(s), abc(2)*sin(s), 'k', 'LineWidth', 1.5);
hold off;
title(sprintf('Re u, plane wave on a sound-hard ellipsoid, k = %g', zk));
xlabel('x');
ylabel('y');

% Ellipsoid with semi-axes abc as a surfer of quad patches: the unit
% cube-sphere of quad_sphere, stretched by diag(abc).
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
