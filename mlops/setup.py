from pathlib import Path

from setuptools import find_namespace_packages, setup

BASE_DIR = Path(__file__).parent
with open(Path(BASE_DIR, "requirements.txt"), "r") as file:
    required_packages = [line.strip() for line in file.readlines()]

setup(
    name="tagiftip",
    version=0.1,
    description="Classifies if a driver will receive a tip or not",
    author="Jagos",
    author_email="jagac41@gmail.com",
    python_requires="= 3.10.6",
    packages=find_namespace_packages(),
    install_requirements=[required_packages],
)
