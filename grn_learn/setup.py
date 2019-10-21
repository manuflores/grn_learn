import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name='grn_learn',
    version='0.0.1',
    author='Emanuel Flores B.',
    author_email='efflores@caltech.edu',
    description='Utilities for machine learning on gene regulatory networks in bacteria.',
    long_description=long_description,
    long_description_content_type='ext/markdown',
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ),
)