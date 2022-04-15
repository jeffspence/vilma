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
    install_requires=['numpy>=1.14.2',
                      'scipy>=1.0.0',
                      'numba>=0.47.0',
                      'pandas>=0.23.4',
                      'tables>=3.3.0',
                      'pytest>=3.0.7'],
    entry_points={
        'console_scripts': ['vilma = vilma.frontend:main'],
    },
)
