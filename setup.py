#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import io
import re
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

from setuptools import find_packages
from setuptools import setup


def read(*names, **kwargs):
    with io.open(join(dirname(__file__), *names), encoding=kwargs.get('encoding', 'utf8')) as fh:
        return fh.read()

extra_seq = ['biopython',
             'pysam']

extra_scseq = ['scanpy',
               'scvi-tools',
               'celltypist',
               'statannot',
               'pybiomart',
               'scrublet',
               'scikit-misc']

extra_dev = [*extra_seq, 
             *extra_scseq]

setup(
    name='kgbio',
    version='0.0.1',
    license='MIT',
    description='My one-stop-shop for bioinformatics utilities that I have written over the years.',
    long_description='{}\n{}'.format(
        re.compile('^.. start-badges.*^.. end-badges', re.M | re.S).sub('', read('README.rst')),
        re.sub(':[a-z]+:`~?(.*?)`', r'``\1``', read('CHANGELOG.rst')),
    ),
    author='Kenneth Gouin III',
    author_email='kgouiniii@gmail.com',
    url='https://github.com/gouink/kgbio',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        'Private :: Do Not Upload',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
    # project_urls={
    #     'Documentation': 'https://kgbio.readthedocs.io/',
    #     'Changelog': 'https://kgbio.readthedocs.io/en/latest/changelog.html',
    #     'Issue Tracker': 'https://github.com/gouink/kgbio/issues',
    # },
    # keywords=[
    #     # eg: 'keyword1', 'keyword2', 'keyword3',
    # ],
    python_requires='>=3.6, <4',
    install_requires=open('requirements.txt').readlines(),
    extras_require={
        'seq': extra_seq,
        'scseq': extra_scseq,
        'dev': extra_dev
    },
    entry_points={
        'console_scripts': [
            'kgbio = kgbio.cli:main',
        ]
    },
)
