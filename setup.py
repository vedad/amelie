import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="amelie", # Replace with your own username
    version="0.0.1",
    author="Vedad Kunovac",
    author_email="vxh710@bham.ac.uk",
    description="Analysis of light curves and radial velocities of transiting
    planets and eclipsing binaries",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vedad/amelie",
    include_package_data=True,  # Checks MANIFEST.in for explicit rules
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
