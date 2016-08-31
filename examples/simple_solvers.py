"""Solve A(x) = b using conjugate gradient."""

import numpy as np
import odl
import spectral_projector

# Discrete reconstruction space: discretized functions on the rectangle
# [-20, 20]^2 with 300 samples per dimension.
reco_space = odl.uniform_discr([-20, -20], [20, 20], [300, 300])

# Make a parallel beam geometry with flat detector
angle_partition = odl.uniform_partition(0, 2 * np.pi, 600)
detector_partition = odl.uniform_partition(-30, 30, 600)
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

# ray transform aka forward projection.
ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl='astra_cuda')

# Scale ray trafo so we have reasonable values
ray_trafo *= 0.3

# Spectrum
sigma = [0.5, 0.5]
mu = [0.5, 1.5]
spectral_ray_trafo = spectral_projector.SpectralProjector(ray_trafo, sigma, mu)

# Create a discrete Shepp-Logan phantom (modified version)
phantom = odl.phantom.shepp_logan(reco_space, modified=True)

# Apply projector
data = spectral_ray_trafo(phantom)

phantom.show('phantom')
data.show('data')

# Callback for solver
callback = (odl.solvers.CallbackShow(display_step=5, clim=[0, 1]) &
            odl.solvers.CallbackShow(display_step=5, coords=[None, 0]) &
            odl.solvers.CallbackPrintIteration())

# Solve using landweber
opnorm = odl.power_method_opnorm(ray_trafo)
x = spectral_ray_trafo.domain.zero()
odl.solvers.landweber(spectral_ray_trafo, x, data, omega=2.0/opnorm**2,
                      niter=100, callback=callback)

# Reset the callback so we get new plots
callback.reset()

# Solve using nonlinear solver
x = spectral_ray_trafo.domain.zero()
for i in range(10):
    odl.solvers.conjugate_gradient_normal(spectral_ray_trafo, x, data,
                                          niter=10, callback=callback)

# Reset the callback so we get new plots
callback.reset()

# Solve using iterative linearization
x = spectral_ray_trafo.domain.zero()
for i in range(5):
    deriv = spectral_ray_trafo.derivative(x)
    dx = spectral_ray_trafo.domain.zero()
    rhs = data - spectral_ray_trafo(x)

    def callback_plus_x(dx):
        callback(x + dx)

    odl.solvers.conjugate_gradient_normal(deriv, dx, rhs, niter=10,
                                          callback=callback_plus_x)
    x += dx
