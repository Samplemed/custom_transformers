import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="custom_transformers",
    version="0.0.1",
    author="Leonardo Uchôa Pedreira",
    author_email="leonardo.pedreira@samplemed.com.br",
    description="Custom Transformers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Samplemed/custom_transformers",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    install_requires=["scikit-learn>=1.0.2"],
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
