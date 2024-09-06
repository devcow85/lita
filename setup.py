from setuptools import setup, find_packages

setup(
    name="lita",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "transformers==4.38",
        "pandas",
        "optimum[onnxruntime]",
        "torchvision",
        "onnxruntime-gpu"
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
        ],
    },
    entry_points={
        "console_scripts": [
            "lita=lita.main:main",
        ],
    },
)