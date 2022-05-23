# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from setuptools import setup, find_packages

setup(
    name='msccl',
    version='2.3.0',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'msccl = msccl.__main__:main',
        ],
    },
    scripts = [
        'msccl/autosynth/msccl_ndv2_launcher.sh'
    ],
    install_requires=[
        'dataclasses; python_version < "3.7"',
        'z3-solver',
        'argcomplete',
        'lxml',
        'humanfriendly',
        'tabulate',
        'igraph'
    ],
    python_requires='>=3.6',
)
