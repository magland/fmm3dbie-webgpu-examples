% Pressure at two listeners near a sound-hard sphere vs frequency, compared
% with the exact series.

mip install --channel mip-org/labs fmm3dbie-webgpu
mip load fmm3dbie-webgpu

a = 1;                                 % sphere radius
xs = [3; 0; 0];                        % point source
xl = [1.8 -1.6; 0.8 0.6; 0 0];         % listeners: in front of and behind the sphere
S = quad_sphere(a, 2, [0; 0; 0], 6);
fprintf('sphere: %d patches, %d points\n', S.npatches, S.npts);

kas = linspace(0.5, 6, 16);
gain = zeros(2, numel(kas));
gain_ex = zeros(2, numel(kas));
t = tic;
for k = 1:numel(kas)
  zk = kas(k)/a;
  G = @(x) exp(1i*zk*vecnorm(x - xs))./(4*pi*vecnorm(x - xs));
  dGdn = @(x, n) sum((x - xs).*n, 1).*(1i*zk*vecnorm(x - xs) - 1) ...
    .*exp(1i*zk*vecnorm(x - xs))./(4*pi*vecnorm(x - xs).^3);
  [dens, ~, ~, Q] = helm3d.neumann.solver(S, -dGdn(S.r, S.n), 1e-6, zk, 1);
  us = helm3d.neumann.eval(S, dens, xl, 1e-6, zk, 1, struct('precomp_quadrature', Q));
  fmm3dbie_gpu('free', Q.gpu_handle);
  for m = 1:2
    gain(m, k) = 20*log10(abs(G(xl(:, m)) + us(m))/abs(G(xl(:, m))));
    gain_ex(m, k) = 20*log10(abs(G(xl(:, m)) + sphere_point_source(zk, a, xs, xl(:, m)))/abs(G(xl(:, m))));
  end
end
fprintf('%d frequencies in %.1f s; max difference from exact %.1e dB\n', ...
  numel(kas), toc(t), max(abs(gain(:) - gain_ex(:))));

figure;
plot(kas, gain(1, :), 'LineWidth', 1.5);
hold on;
plot(kas, gain_ex(1, :), 'ko');
plot(kas, gain(2, :), 'LineWidth', 1.5);
plot(kas, gain_ex(2, :), 'ko');
hold off;
xlabel('ka');
ylabel('level relative to free field (dB)');
legend('in front, GPU', 'in front, exact', 'behind, GPU', 'behind, exact');
title('Listeners near a sound-hard sphere');

% Exact field scattered by a sound-hard sphere of radius a at the origin
% from a point source at xs, evaluated at x (both outside the sphere).
function us = sphere_point_source(zk, a, xs, x)
  c = dot(x, xs)/(norm(x)*norm(xs));
  nmax = ceil(zk*max(norm(x), norm(xs)) + 20);
  us = 0;
  P0 = 1;
  P1 = c;
  for n = 0:nmax
    if n == 0
      P = P0;
    elseif n == 1
      P = P1;
    else
      P = ((2*n - 1)*c*P1 - (n - 1)*P0)/n;
      P0 = P1;
      P1 = P;
    end
    [~, dja] = sph_j(n, zk*a);
    [~, dha] = sph_h(n, zk*a);
    us = us - 1i*zk/(4*pi)*(2*n + 1)*dja/dha*sph_h(n, zk*norm(x))*sph_h(n, zk*norm(xs))*P;
  end
end

function [j, dj] = sph_j(n, z)
  j = sqrt(pi/(2*z))*besselj(n + 0.5, z);
  dj = n/z*j - sqrt(pi/(2*z))*besselj(n + 1.5, z);
end

function [h, dh] = sph_h(n, z)
  f = @(m) sqrt(pi/(2*z))*(besselj(m + 0.5, z) + 1i*bessely(m + 0.5, z));
  h = f(n);
  dh = n/z*h - f(n + 1);
end
