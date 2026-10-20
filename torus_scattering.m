% Plane wave on a sound-hard torus, after a point-source self-test.

mip install --channel mip-org/labs fmm3dbie-webgpu
mip load fmm3dbie-webgpu

R = 1.2;   % major radius
r = 0.5;   % minor radius
zk = 5;
S = quad_torus(R, r, 12, 6, 6);
fprintf('torus: %d patches, %d points\n', S.npatches, S.npts);

% 1. Self-test: a source inside the tube gives an exterior solution.
x0 = [R + 0.1; 0.05; -0.1];
G = @(x) exp(1i*zk*vecnorm(x - x0))./(4*pi*vecnorm(x - x0));
dGdn = @(x, n) sum((x - x0).*n, 1).*(1i*zk*vecnorm(x - x0) - 1) ...
  .*exp(1i*zk*vecnorm(x - x0))./(4*pi*vecnorm(x - x0).^3);
t = tic;
[dens, errs, ~, Q] = helm3d.neumann.solver(S, dGdn(S.r, S.n), 1e-6, zk, 1);
fprintf('self-test solve: %d GMRES iterations, %.1f s\n', numel(errs), toc(t));
chk = [0 0 0; 0 0 1.5; -3 0.5 0.2; 1 2.5 -0.7].';
p = helm3d.neumann.eval(S, dens, chk, 1e-6, zk, 1, struct('precomp_quadrature', Q));
fprintf('self-test: max relative error %.1e (one check point is in the hole)\n', ...
  max(abs(p.' - G(chk)))/max(abs(G(chk))));

% 2. Plane wave from above at an angle
d = [0.5; 0; -1]/norm([0.5; 0; -1]);
uinc = @(x) exp(1i*zk*(d.'*x));
rhs = -1i*zk*(d.'*S.n).*uinc(S.r);
t = tic;
[dens, errs] = helm3d.neumann.solver(S, rhs, 1e-6, zk, 1, struct('quadrature_correction', Q));
fprintf('plane wave solve: %d GMRES iterations, %.1f s\n', numel(errs), toc(t));

% Field in the plane y = 0, which cuts through the tube twice
n = 141;
L = 3;
[X, Z] = meshgrid(linspace(-L, L, n));
targ = [X(:).'; zeros(1, n^2); Z(:).'];
out = (abs(X(:).') - R).^2 + Z(:).'.^2 > (1.03*r)^2;
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
s = linspace(0, 2*pi, 100);
plot(R + r*cos(s), r*sin(s), 'k', 'LineWidth', 1.5);
plot(-R + r*cos(s), r*sin(s), 'k', 'LineWidth', 1.5);
hold off;
title(sprintf('Re u at y = 0, sound-hard torus, k = %g', zk));
xlabel('x');
ylabel('z');

% Torus with major radius R and minor radius r as a surfer of quad patches:
% nu x nv patches in the angles (u around the axis, v around the tube).
function S = quad_torus(R, r, nu, nv, norder)
  st = polytens.lege.nodes(norder);
  npols = size(st, 2);
  srcvals = zeros(12, nu*nv*npols);
  hu = 2*pi/nu;
  hv = 2*pi/nv;
  ip = 0;
  for i = 1:nu
    for j = 1:nv
      u = (i - 1)*hu + (st(1, :) + 1)*hu/2;
      v = (j - 1)*hv + (st(2, :) + 1)*hv/2;
      rho = R + r*cos(v);
      x = [rho.*cos(u); rho.*sin(u); r*sin(v)];
      xu = [-rho.*sin(u); rho.*cos(u); zeros(size(u))]*(hu/2);
      xv = [-r*sin(v).*cos(u); -r*sin(v).*sin(u); r*cos(v)]*(hv/2);
      nrm = [cos(v).*cos(u); cos(v).*sin(u); sin(v)];
      srcvals(:, ip*npols + (1:npols)) = [x; xu; xv; nrm];
      ip = ip + 1;
    end
  end
  S = surfer(nu*nv, norder, srcvals, 11);
end
