% Scattering patterns of a sphere with impedance lam = 0, 0.5, 1, vs exact.

mip install --channel mip-org/labs fmm3dbie-webgpu
mip load fmm3dbie-webgpu

a = 1;          % sphere radius
zk = 5;         % wavenumber (ka = 5)
S = quad_sphere(a, 3, [0; 0; 0], 6);
fprintf('sphere: %d patches, %d points\n', S.npatches, S.npts);

% Plane wave along +z; targets on a circle of radius R in the xz-plane
d = [0; 0; 1];
uinc = @(x) exp(1i*zk*(d.'*x));
duinc_dn = @(x, n) 1i*zk*(d.'*n).*uinc(x);
R = 10;
th = linspace(0, pi, 181);
targ = R*[sin(th); zeros(size(th)); cos(th)];

lams = [0 0.5 1];
Q = [];
figure;
hold on;
for k = 1:numel(lams)
  lam = lams(k);
  rhs = -(duinc_dn(S.r, S.n) + 1i*zk*lam*uinc(S.r));
  t = tic;
  if isempty(Q)
    [dens, errs, ~, Q] = helm3d.impedance.solver(S, lam, rhs, 1e-6, zk, 1);
  else
    [dens, errs] = helm3d.impedance.solver(S, lam, rhs, 1e-6, zk, 1, ...
      struct('quadrature_correction', Q));
  end
  tsolve = toc(t);
  us = helm3d.impedance.eval(S, dens, targ, 1e-6, zk, 1, struct('precomp_quadrature', Q));
  uex = mie_impedance(zk, a, lam, targ);
  fprintf('lam = %.1f: %2d GMRES iterations, solve %.1f s, max error vs exact %.1e\n', ...
    lam, numel(errs), tsolve, max(abs(us.' - uex))/max(abs(uex)));
  plot(th*180/pi, R*abs(us), 'LineWidth', 1.5);
  plot(th(1:10:end)*180/pi, R*abs(uex(1:10:end)), 'ko');
end
hold off;
fmm3dbie_gpu('free', Q.gpu_handle);
xlabel('angle from the direction of incidence (degrees)');
ylabel('R |u_s|');
title(sprintf('Scattering pattern, ka = %g (lines: GPU, circles: exact)', zk*a));
legend('GPU, lambda = 0', 'exact, lambda = 0', 'GPU, lambda = 0.5', 'exact, lambda = 0.5', ...
  'GPU, lambda = 1', 'exact, lambda = 1');

% Exact scattered field of the plane wave exp(1i*zk*z) by a sphere of radius
% a at the origin with du/dn + 1i*zk*lam*u = 0, at points x outside it.
function us = mie_impedance(zk, a, lam, x)
  r = vecnorm(x);
  c = x(3, :)./r;
  nmax = ceil(zk*a + 4*(zk*a)^(1/3) + 10);
  us = zeros(size(r));
  P0 = ones(size(c));
  P1 = c;
  for n = 0:nmax
    if n == 0
      P = P0;
    elseif n == 1
      P = P1;
    else
      P = ((2*n - 1)*c.*P1 - (n - 1)*P0)/n;
      P0 = P1;
      P1 = P;
    end
    [ja, dja] = sph_j(n, zk*a);
    [ha, dha] = sph_h(n, zk*a);
    coef = -(1i^n)*(2*n + 1)*(dja + 1i*lam*ja)/(dha + 1i*lam*ha);
    us = us + coef*sph_h(n, zk*r).*P;
  end
end

function [j, dj] = sph_j(n, z)
  j = sqrt(pi./(2*z)).*besselj(n + 0.5, z);
  dj = n./z.*j - sqrt(pi./(2*z)).*besselj(n + 1.5, z);
end

function [h, dh] = sph_h(n, z)
  f = @(m) sqrt(pi./(2*z)).*(besselj(m + 0.5, z) + 1i*bessely(m + 0.5, z));
  h = f(n);
  dh = n./z.*h - f(n + 1);
end
