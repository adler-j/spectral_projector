import numpy as np
import odl


__all__ = ('SpectralProjector',)


class SpectralProjector(odl.Operator):
    """Projector for spectral CT.

    Notes
    -----

    Computes

    .. math::

       f \\rightarrow \\int \\sigma(E) e^{- \\mu(E) A(f)} dE

    Where:

    * :math:`\\sigma(E)` is the spectrum at energy :math:`E`.
    * :math:`\\mu(E)` is the mass attenuation coeffcient at eneryg
      :math:`E`
    * :math:`A(f)` is the ray transform of data :math:`f`.
    """

    def __init__(self, ray_transform, sigma, mu):
        """init

        Parameters
        ----------
        ray_transform : `odl.Operator`
            The forward model to use, intended to be a `odl.tomo.RayTransform`.
        sigma : `array-like`
            The weighting of each part of the spectrum.
        mu : `array-like`
            Attenuation at each energy. Same size as ``spectrum``

        Examples
        --------
        Create a ODL `RayTransform`

        >>> import odl
        >>> space = odl.uniform_discr([-1, -1], [1, 1], [50, 50])
        >>> apart = odl.uniform_partition(0, 3.1415, 100)
        >>> dpart = odl.uniform_partition(-2, 2, 100)
        >>> geom = odl.tomo.Parallel2dGeometry(apart, dpart)
        >>> ray_trafo = odl.tomo.RayTransform(space, geom)

        Create the `SpectralProjector`

        >>> sigma = [0.5, 0.5]
        >>> mu = [0.5, 1.5]
        >>> SpectralProjector(ray_trafo, sigma, mu)
        """
        odl.tomo.RayTransform
        self.ray_transform = ray_transform
        self.mu = np.atleast_1d(mu)
        self.sigma = np.atleast_1d(sigma)
        odl.Operator.__init__(self, ray_transform.domain, ray_transform.range,
                              linear=False)

    def _call(self, x, **kwargs):

        # Optimization to avoid double computations in derivative
        if 'Ahatrho' in kwargs:
            Ahatrho = kwargs.pop('Ahatrho')
        else:
            Ahatrho = self.ray_transform(x)

        result = self.range.zero()
        for s, m in zip(self.sigma, self.mu):
            result += s * np.exp(- m * Ahatrho)

        return -np.log(result)

    def derivative(self, point):
        """Derivative w.r.t. the data."""
        Ahatrho = self.ray_transform(point)
        Arho = self(point, Ahatrho=Ahatrho)

        result = self.range.zero()
        for s, m in zip(self.sigma, self.mu):
            result += (m * s) * np.exp(Arho - m * Ahatrho)

        return result * self.ray_transform
