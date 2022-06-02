import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="custom_transformers",
    version="0.0.1",
    author="Leonardo Uchôa Pedreira",
    author_email="leonardo.pedreira@samplemed.com.br",
    description="Custom Transformers",
    long_description=long_description,
    url="https://github.com/Samplemed/preditivo_vida",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)


