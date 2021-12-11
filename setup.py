from setuptools import setup, find_packages
from distutils.core import setup

with open("README.md", "r") as rm:
    concise_intro = rm.read()

setup(
    name = 'AutoDiffRunTimeError',
    version = '0.0.2',
    license='MIT',
    packages=find_packages(),
    description = 'Automatic Differentiation Package',
    author = 'RunTimeTerror',
    url = 'https://github.com/cs107-runtimeterror/cs107-FinalProject',
    long_description = concise_intro,
    long_description_content_type = 'text/markdown',
    keywords = ['Automatic Differentiation', 'AD'],
    install_requires=[
        'numpy'
    ],
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    setup_requires=['wheel'],
    python_requires = '>=3.6',
)
