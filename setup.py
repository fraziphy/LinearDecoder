# setup.py

from setuptools import setup, find_packages
import os

# User-friendly description from README.md
current_directory = os.path.dirname(os.path.abspath(__file__))
try:
    with open(os.path.join(current_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except IOError:
    long_description = ''

setup(
    # Name of the package
    name='lineardecoder',

    # Packages to include into the distribution
    packages=find_packages('src'),  # Find packages under the 'src' directory
    package_dir={'': 'src'},  # Tell setuptools packages are under 'src'

    # Start with a small number and increase it with every change you make
    # https://semver.org
    version='0.1.0',

    # Chose a license from here: https://help.github.com/articles/licensing-a-repository
    license='GNU General Public License v3 (GPLv3)',

    # Short description of your library
    description='A module for reconstructing external stimuli from evoked neural responses using machine learning.',

    # Long description of your library
    long_description=long_description,
    long_description_content_type='text/markdown',

    # Your name
    author='Farhad Razi',

    # Your email
    author_email='farhad.razi.1988@gmail.com',

    # Either the link to your github or to your website
    url='https://github.com/fraziphy/LinearDecoder',  # Correct the repo name

    # Link from which the project can be downloaded
    download_url='https://github.com/fraziphy/LinearDecoder/archive/refs/tags/v0.1.0.tar.gz',

    # List of keyword arguments
    keywords=['neural decoding', 'machine learning', 'neuroscience', 'stimulus reconstruction'],

    # List of packages to install with this one
    install_requires=[
        'numpy',
        'matplotlib', # Consider adding common dependencies
        'scikit-learn'
    ],

    # https://pypi.org/classifiers/
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',  # Correct the License
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Neuroscience'
    ],
    python_requires='>=3.7'
)
