========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |github-actions| |requires|
        | |codecov|
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|
.. |docs| image:: https://readthedocs.org/projects/kgbio/badge/?style=flat
    :target: https://kgbio.readthedocs.io/
    :alt: Documentation Status

.. |github-actions| image:: https://github.com/gouink/kgbio/actions/workflows/github-actions.yml/badge.svg
    :alt: GitHub Actions Build Status
    :target: https://github.com/gouink/kgbio/actions

.. |requires| image:: https://requires.io/github/gouink/kgbio/requirements.svg?branch=main
    :alt: Requirements Status
    :target: https://requires.io/github/gouink/kgbio/requirements/?branch=main

.. |codecov| image:: https://codecov.io/gh/gouink/kgbio/branch/main/graphs/badge.svg?branch=main
    :alt: Coverage Status
    :target: https://codecov.io/github/gouink/kgbio

.. |version| image:: https://img.shields.io/pypi/v/kgbio.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/kgbio

.. |wheel| image:: https://img.shields.io/pypi/wheel/kgbio.svg
    :alt: PyPI Wheel
    :target: https://pypi.org/project/kgbio

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/kgbio.svg
    :alt: Supported versions
    :target: https://pypi.org/project/kgbio

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/kgbio.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/kgbio

.. |commits-since| image:: https://img.shields.io/github/commits-since/gouink/kgbio/v0.0.2.svg
    :alt: Commits since latest release
    :target: https://github.com/gouink/kgbio/compare/v0.0.2...main



.. end-badges

My one-stop-shop for bioinformatics utilities that I have written over the years.

* Free software: MIT license

Installation
============

::

    ## Clone the repo
    git clone git@github.com:gouinK/kgbio.git
    
    ## Navigate to repo
    cd kgbio

    ## To install just utils and helpers
    pip install .

    ## To install seq module
    pip install .[seq]

    ## To install scseq module
    pip install .[scseq]

    ## To install everything
    pip install .[seq,scseq]

Documentation
=============

The utils module contains plotting helper functions.

The seq module contains functions related to fastq processing.

The scseq module contains functions related to single-cell-sequencing processing.

utils 
    -> plotting.py

seq
    -> rawdata.py

scseq
    -> generalfunctions.py

    -> qcfunctions.py

    -> dgexfunctions.py

    -> populationfunctions.py (work in progress, no guarantees)

    -> airrfunctions.py (work in progress, no guarantees)

Not currently active: https://kgbio.readthedocs.io/

Development
===========

Not currently implemented:

To run all the tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
