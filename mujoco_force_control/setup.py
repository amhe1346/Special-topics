from setuptools import setup, find_packages

setup(
    name="mujoco-force-control",
    version="0.1.0",
    description="MuJoCo learning project: from PD control to π₀.₅ VLA policy learning",
    author="HIRO Lab",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "mujoco>=3.1.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "h5py>=3.9.0",
    ],
)
