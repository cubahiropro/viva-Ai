from setuptools import find_packages, setup

setup(
    name="viva_ai",
    version="0.1.0",
    description="On-device personal-life intelligence training pipeline for the Viva app.",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
)
