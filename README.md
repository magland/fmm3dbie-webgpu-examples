# fmm3dbie on WebGPU in numbl

Acoustic scattering with [fmm3dbie](https://github.com/fastalgorithms/fmm3dbie)'s `helm3d` solvers, running on the GPU via the `fmm3dbie-webgpu` package. Needs a browser with WebGPU (recent Chrome or Edge). Open a script and press Run.

**Live:** https://magland.github.io/fmm3dbie-webgpu-examples/

- `sphere_scattering.m`: plane wave on a sound-hard sphere
- `absorbing_sphere.m`: scattering patterns for three boundary impedances
- `ellipsoid_scattering.m`: plane wave on an ellipsoid
- `two_spheres.m`: plane wave on two spheres
- `torus_scattering.m`: plane wave on a torus
- `room_interior.m`: point source in a room with absorbing walls
- `frequency_response.m`: level at two listeners near a sphere vs frequency (about a minute)
