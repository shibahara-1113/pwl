import sys
import warnings
from setuptools import setup, find_packages

if "develop" not in sys.argv:
    warnings.warn("since Point-Wise Linear (PWL) is under rapid, active "
                  "development, `python setup.py install` is "
                  "intentionally disabled to prevent other "
                  "problems. Run `python setup.py develop` to "
                  "install PWL.")

setup(
    name="PWL",
    version="2023.01.16",
    packages=find_packages(),
    description="A deep learning library",
    install_requires=["torch"],
    test_suite="nose.collector",
)
