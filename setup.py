import setuptools
import os

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="genderperformr",
    version="1.2",
    author="Zijian Wang and David Jurgens",
    author_email="zijwang@stanford.edu",
    description="GenderPerformr",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.6",
    install_requires=["torch>=0.4.1", "numpy", "unidecode"],
    url="https://github.com/zijwang/genderperformr",
    include_package_data=True,
    packages=setuptools.find_packages()

)