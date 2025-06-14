from setuptools import setup
import version

setup(
    name='sbk-charts',
    version=version.__version__,
    packages=['charts'],
    url='https://github.com/kmgowda/sbk-charts',
    license='Apache License Version 2.0',
    author='KMG',
    author_email='keshava.gowda@gmail.com',
    description='SBK Charts',
    executables = [{"script": "sbk-charts"}],
)
