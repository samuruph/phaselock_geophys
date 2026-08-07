from setuptools import find_packages, setup

with open("README.md") as handle:
    long_description = handle.read()

setup(
    name="phaselock",
    version="0.2.0",
    description="Trajectory geometry in video diffusion internals: PhaseLock + GeoPhys",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "diffusers>=0.39.0",
        "transformers>=4.35.0",
        "accelerate>=0.25.0",
        "torchvision>=0.15.0",
        "opencv-python>=4.8.0",
        "pillow>=9.0.0",
        "numpy>=1.21.0",
        "pyyaml>=6.0",
    ],
    extras_require={"dev": ["pytest", "black", "isort"]},
    license="Apache-2.0",
)
