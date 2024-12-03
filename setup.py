from os.path import dirname, join
from pkg_resources import parse_requirements
from setuptools import setup, find_packages

with open("README.md") as file:
    long_description = file.read()

setup(
    name="timbre",
    py_modules=["timbre"],
    version="0.0",
    description="",
    author="Quinn Ouyang",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        str(r)
        for r in parse_requirements(open(join(dirname(__file__), "requirements.txt")))
    ],
    include_package_data=True,
    author_email="qouyang3@illinois.edu",
    url="https://github.com/quinnouyang/timbre",
    package_data={"timbre": ["assets/*", "assets/*/*"]},
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=[],
    classifiers=["License :: OSI Approved :: MIT License"],
    license="MIT",
)
