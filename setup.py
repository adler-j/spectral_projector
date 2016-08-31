# Copyright 2014-2016 The ODL development group
#
# This file is part of ODL.
#
# ODL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ODL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ODL.  If not, see <http://www.gnu.org/licenses/>.

"""Setup script for spectral_projector.

Installation command::

    pip install [--user] [-e] .
"""

from __future__ import print_function, absolute_import

from setuptools import setup, find_packages

setup(
    name='spectral_projector',

    version='alpha',

    url='https://github.com/adler-j/spectral_projector',

    author='Jonas Adler',
    author_email='jonasadl@kth.se',

    license='GPLv3+',

    packages=find_packages(exclude=['*test*']),
    package_dir={'spectral_projector': 'spectral_projector'},

    install_requires=['odl'],
    tests_require=['pytest'],
)
