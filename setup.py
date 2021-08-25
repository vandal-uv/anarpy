import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="anarpy",
    version="0.1.8",
    author="Valparaiso Neural Dynamics Laboratory",
    author_email="javier.palma@cinv.cl",
    description="Analysis and Replication in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://anarpy.readthedocs.io",
    project_urls={
        "Bug Tracker": "https://github.com/vandal-uv/anarpy/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src/"),
    python_requires=">=3.6",
)