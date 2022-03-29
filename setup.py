# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from setuptools import setup, find_packages

setup(
    name='sccl',
    version='2.3.0',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'sccl = sccl.__main__:main',
        ],
    },
    scripts = [
        'sccl/autosynth/sccl_ndv2_launcher.sh'
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
