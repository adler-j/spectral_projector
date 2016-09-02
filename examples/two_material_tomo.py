"""Solve A(x) = b using conjugate gradient."""

import numpy as np
import odl
from spectral_projector import SpectralProjector

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
ray_trafo *= 0.4

# Spectrum
sigma = [0.5, 0.5]

# Bone trafo
mu = [4.0, 0.5]
bone_ray_trafo = SpectralProjector(ray_trafo, sigma, mu)

# Water trafo
mu = [1.0, 1.0]
water_ray_trafo = SpectralProjector(ray_trafo, sigma, mu)

# Total trafo
spectral_ray_trafo = odl.ReductionOperator(bone_ray_trafo, water_ray_trafo)

# Create a discrete Shepp-Logan phantom (modified version)
shepp_logan = odl.phantom.shepp_logan(reco_space, modified=True)
water_mask = np.logical_and(np.greater(shepp_logan, 0.1),
                            np.less(shepp_logan, 0.5))
phantom = spectral_ray_trafo.domain.element([(1 - water_mask) * shepp_logan,
                                             water_mask * shepp_logan])
phantom.show('phantom')

# Apply projector
data = spectral_ray_trafo(phantom)
data.show('data')

# Callback for solver
callback = (odl.solvers.CallbackShow(display_step=5, clim=[0, 1]) &
            odl.solvers.CallbackShow(display_step=5, coords=[None, 0]) &
            odl.solvers.CallbackPrintIteration())
"""
# Solve using landweber
opnorm = odl.power_method_opnorm(ray_trafo)
x = spectral_ray_trafo.domain.zero()
odl.solvers.landweber(spectral_ray_trafo, x, data, omega=1.0/opnorm**2,
                      niter=200, callback=callback)

# Solve using tikhonov regularized iterative linearization
callback.reset()


# Solve using iterative linearization
A = spectral_ray_trafo
B = 0.1 * odl.IdentityOperator(spectral_ray_trafo.domain)
x = spectral_ray_trafo.domain.zero()
for i in range(50):
    deriv = A.derivative(x)
    dx = A.domain.zero()
    rhs = deriv.adjoint(data - A(x)) - B.adjoint(B(x))
    op = deriv.adjoint * deriv + B.adjoint * B

    def callback_plus_x(dx):
        callback(x + dx)

    odl.solvers.conjugate_gradient(op, dx, rhs, niter=20,
                                   callback=callback_plus_x)
    x += dx
"""
# Using douglas rachford

# Get step lengths
opnorm_A = odl.power_method_opnorm(2 * ray_trafo, xstart=shepp_logan)
sigma = [1.0 / opnorm_A**2, 1.0]
tau = 1.0

A = spectral_ray_trafo
B = odl.IdentityOperator(spectral_ray_trafo.domain)
x = A.domain.zero()
for i in range(10):
    deriv = A.derivative(x)
    lin_ops = [deriv, B]

    prox_cc_g = [odl.solvers.proximal_cconj_l2(spectral_ray_trafo.range, g=data - A(x)),
                 odl.solvers.proximal_cconj_l1(spectral_ray_trafo.domain, lam=0.001, g=-x)]

    # Proximal of the bound constraint 0 <= x + dx <= infty
    prox_f = odl.solvers.proximal_box_constraint(spectral_ray_trafo.domain, lower=-x)

    def callback_plus_x(dx):
        callback(x + dx)

    # Solve using the Douglas-Rachford Primal-Dual method
    dx = A.domain.zero()
    odl.solvers.douglas_rachford_pd(dx, prox_f, prox_cc_g, lin_ops,
                                    tau=tau, sigma=sigma, niter=30,
                                    callback=callback_plus_x)

    x += dx
