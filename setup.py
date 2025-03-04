<<<<<<< HEAD
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="anarpy",
    version="0.3.0",
    author="Valparaiso Neural Dynamics Laboratory",
    author_email="javier.palma@cinv.cl",
    description="Analysis and Replication in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://anarpy.readthedocs.io",
    project_urls={
        "Bug Tracker": "https://github.com/vandal-uv/anarpy/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
=======
import sys
from os import path

from setuptools import find_packages, setup

import versioneer

# NOTE: This file must remain Python 2 compatible for the foreseeable future,
# to ensure that we error out properly for people with outdated setuptools
# and/or pip.
min_version = (
    3,
    9,
)
if sys.version_info < min_version:
    error = """
anarpy does not support Python {0}.{1}.
Python {2}.{3} and above is required. Check your Python version like so:

python3 --version

This may be due to an out-of-date pip. Make sure you have pip >= 9.0.1.
Upgrade pip like so:

pip install --upgrade pip
""".format(
        *(sys.version_info[:2] + min_version)
    )
    sys.exit(error)

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.rst"), encoding="utf-8") as readme_file:
    readme = readme_file.read()

with open(path.join(here, "requirements.txt")) as requirements_file:
    # Parse requirements.txt, ignoring any commented-out lines.
    requirements = [line for line in requirements_file.read().splitlines() if not line.startswith("#")]


setup(
    name="anarpy",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="AnarPy is a Python package to facilitate the simulation, analysis, and replication of several experiments using computational whole brain models. For more details, installation instructions, documentation, tutorials, forums, videos and more, please visit: https://anarpy.readthedocs.io",
    long_description=readme,
    author="Vandal",
    author_email="javier.palma@cinv.cl",
    url="https://github.com/jpalma-espinosa/anarpy",
    python_requires=">={}".format(".".join(str(n) for n in min_version)),
    packages=find_packages(exclude=["docs", "tests"]),
    entry_points={
        "console_scripts": [
            # 'command = some.module:some_function',
        ],
    },
    include_package_data=True,
    package_data={
        "anarpy": [
            # When adding files here, remember to update MANIFEST.in as well,
            # or else they will not be included in the distribution on PyPI!
            # 'path/to/data_file',
        ]
    },
    install_requires=requirements,
    license="BSD (3-clause)",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
    ],
)
>>>>>>> 15ae086 (File and folder structure for compliance with cookiecutter)
