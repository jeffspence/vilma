from src.vilma import VERSION
import setuptools


setuptools.setup(
    name='vilma',
    version=VERSION,
    description='Construct polygenic scores and infer distributions of '
                'effect sizes using variational inference.',
    author='Jeffrey P. Spence, Nasa Sinnott-Armstrong, Themistocles L. '
           'Assimes, Jonathan K. Pritchard',
    author_email='jspence@stanford.edu',
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where='src'),
    install_requires=['numpy>=1.20.0',
                      'plinkio>=0.9.8',
                      'pandas>=1.2.1',
                      'h5py>=3.6.0',
                      'numba==0.55.2'],
    entry_points={
        'console_scripts': ['vilma = vilma.frontend:main'],
    },
)
