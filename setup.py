from setuptools import setup, find_packages

setup(
    name="zf_pf_diffeo",
    version="0.1.0",
    description="Analyses for the zebrafish pectoral fin geometry.",
    author="Maximilian Kotz",
    author_email="maximilian.kotz@tu-dresden.de",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        # Add dependencies here
        # e.g., 'numpy', 'scipy'
    ],
    python_requires='>=3.9',
)