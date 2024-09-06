from setuptools import setup, find_packages

setup(
    name="lita",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "transformers",
        "pandas",
        "optimum[onnxruntime]"
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