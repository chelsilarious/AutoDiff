from distutils.core import setup

with open("README.md", "r") as rm:
    concise_intro = rm.read()

setup(
    name = 'AutoDiff-RunTimeTerror',
    version = '0.0.1',
    license='MIT',
    description = 'Automatic Differentiation Package',
    author = 'RunTimeTerror',
    url = 'https://github.com/cs107-runtimeterror/cs107-FinalProject',
    long_description = concise_intro,
    long_description_content_type = 'text/markdown',
    keywords = ['Automatic Differentiation', 'AD'],
    install_requires=[
        'numpy',
        'inspect',
    ],
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
  python_requires = '>=3.6',
)
