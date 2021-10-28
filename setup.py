import os
from setuptools import setup


def read_requirements():
    """Read the list of required packages from 'requirements.txt', which is
    assumed to be in the same directory as this script."""
    req_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 "requirements.txt")
    with open(req_file_path, 'r') as req_file:
        return list(line.strip() for line in req_file)


setup(
    name='SpectralElementMethod',
    version='0.1.0',
    author='Nicholas G. Chisholm',
    author_email='nchishol@alumni.cmu.edu',
    packages=['sem'],
    description=('A library for solving partial differential equations'
                 'using the spectral element method'),
    install_requires=read_requirements()
)
