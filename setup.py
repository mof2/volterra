#!/usr/bin/env python

from setuptools import setup, Extension
import numpy as np

# read in version number from preg_version module
from preg_version import __version__

setup(name='preg',
    version=__version__,
    description='Gaussian process regression using a polynomial covariance function',
    author='Matthias O. Franz',
    author_email='mfranz@htwg-konstanz.de',
	url='https://github.com/mof2/volterra',
	py_modules=['preg', 'preg_version'],
	setup_requires=['numpy'],
	zip_safe=False,
	provides=['preg']
	)
