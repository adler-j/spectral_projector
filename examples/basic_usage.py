"""Basic usage of the spectral projector."""

import numpy as np
import odl
import spectral_projector

# Discrete reconstruction space: discretized functions on the rectangle
# [-20, 20]^2 with 300 samples per dimension.
reco_space = odl.uniform_discr(
    min_pt=[-20, -20], max_pt=[20, 20], shape=[300, 300], dtype='float64')

# Make a parallel beam geometry with flat detector
# Angles: uniformly spaced, n = 360, min = 0, max = 2 * pi
angle_partition = odl.uniform_partition(0, 2 * np.pi, 360)

# Detector: uniformly sampled, n = 558, min = -30, max = 30
detector_partition = odl.uniform_partition(-30, 30, 558)
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

# ray transform aka forward projection. We use ASTRA CUDA backend.
ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl='astra_cuda')

# Scale ray trafo so we have reasonable values
ray_trafo *= 0.3

# Spectrum
sigma = [0.5, 0.5]
mu = [0.3, 1.7]
spectral_ray_trafo = spectral_projector.SpectralProjector(ray_trafo, sigma, mu)

# Create a discrete Shepp-Logan phantom (modified version)
phantom = odl.phantom.shepp_logan(reco_space, modified=True)
phantom.show('phantom')

# Apply projector
data = spectral_ray_trafo(phantom)
data.show('data')

# Take derivative of projector in the point `phantom` and in the direction `1`
derivative = spectral_ray_trafo.derivative(phantom)
ones = spectral_ray_trafo.domain.one()
deriv_ones = derivative(ones)
deriv_ones.show('data')
