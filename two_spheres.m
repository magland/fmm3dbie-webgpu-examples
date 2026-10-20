% Plane wave on two sound-hard spheres (multiple scattering).

mip install --channel mip-org/labs fmm3dbie-webgpu
mip load fmm3dbie-webgpu

c1 = [-1.3; 0.4; 0];  a1 = 1;
c2 = [1.4; -0.5; 0];  a2 = 0.7;
zk = 5;
S1 = quad_sphere(a1, 3, c1, 6);
S2 = quad_sphere(a2, 2, c2, 6);
S = merge([S1, S2]);
fprintf('two spheres: %d patches, %d points\n', S.npatches, S.npts);

% 1. Self-test: sources inside each sphere give an exterior solution.
x1 = c1 + [0.2; -0.1; 0.1];
x2 = c2 + [-0.1; 0.2; 0];
G = @(x, x0) exp(1i*zk*vecnorm(x - x0))./(4*pi*vecnorm(x - x0));
dGdn = @(x, n, x0) sum((x - x0).*n, 1).*(1i*zk*vecnorm(x - x0) - 1) ...
  .*exp(1i*zk*vecnorm(x - x0))./(4*pi*vecnorm(x - x0).^3);
uref = @(x) G(x, x1) + G(x, x2);
t = tic;
[dens, errs, ~, Q] = helm3d.neumann.solver(S, dGdn(S.r, S.n, x1) + dGdn(S.r, S.n, x2), 1e-6, zk, 1);
fprintf('self-test solve: %d GMRES iterations, %.1f s\n', numel(errs), toc(t));
chk = [0 2 0; 0 -2 0.5; 3 1 -1; -3 -1 1].';
p = helm3d.neumann.eval(S, dens, chk, 1e-6, zk, 1, struct('precomp_quadrature', Q));
fprintf('self-test: max relative error %.1e\n', max(abs(p.' - uref(chk)))/max(abs(uref(chk))));

% 2. Plane wave
d = [1; 0.3; 0]/norm([1; 0.3; 0]);
uinc = @(x) exp(1i*zk*(d.'*x));
rhs = -1i*zk*(d.'*S.n).*uinc(S.r);
t = tic;
[dens, errs] = helm3d.neumann.solver(S, rhs, 1e-6, zk, 1, struct('quadrature_correction', Q));
fprintf('plane wave solve: %d GMRES iterations, %.1f s\n', numel(errs), toc(t));

n = 141;
L = 3.5;
[X, Y] = meshgrid(linspace(-L, L, n));
targ = [X(:).'; Y(:).'; zeros(1, n^2)];
out = vecnorm(targ - c1) > 1.02*a1 & vecnorm(targ - c2) > 1.02*a2;
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
plot(c1(1) + a1*cos(s), c1(2) + a1*sin(s), 'k', 'LineWidth', 1.5);
plot(c2(1) + a2*cos(s), c2(2) + a2*sin(s), 'k', 'LineWidth', 1.5);
hold off;
title(sprintf('Re u, plane wave on two sound-hard spheres, k = %g', zk));
xlabel('x');
ylabel('y');
