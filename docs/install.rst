.. _install:

Installation
=======================================


Requirements 
------------------------------------------------------

Before installing AnarPy, please make sure that your are running Python 3 (3.6 or higher) and ``pip``

1) For a proper Python installation, please download it from the `official Python website <https://www.python.org/>`_. Alternatively, you can download the `Anaconda Distribution <https://www.anaconda.com/distribution/>`_ which also includes several data science and visualization packages.

2) The ``pip`` tool for installing Python packages. See `pip installation here <https://pip.pypa.io/en/stable/installing/>`_.


Install the latest released version of NetPyNE via pip (Recommended)
------------------------------------------------------

Linux or Mac OS:  ``pip install anarpy`` 

Windows: ``python -m pip install anarpy``


Upgrade to the latest released version of AnarPy via pip
------------------------------------------------------

Use this option if you already have AnarPy installed and just want to update to the latest version.

Linux or Mac OS: ``pip install anarpy -U``

Windows: ``python -m pip install -U anarpy`` 


Wanna contribute? (WIP)
------------------------------------------------------

If you want to take part in enhancing AnarPy, we strongly suggest to download the development version of AnarPy via GitHub and pip
The following instructions will install the version in the GitHub "development" branch -- it will include some of the latest enhancements, bug fixes, and new bugs =)

1) ``git clone https://github.com/vandal-uv/anarpy.git``
2) ``cd anarpy``
3) ``git checkout development``
4) ``pip install -e .``

This version can also be used by developers interested in extending the package. 
