from setuptools import setup
from src.charts.version import __version__

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='sbk-charts',
    version=__version__,
    packages=['charts'],
    url='https://github.com/kmgowda/sbk-charts',
    license='Apache License Version 2.0',
    author='KMG',
    author_email='keshava.gowda@gmail.com',
    description='SBK Charts',
    install_requires=required,
)
