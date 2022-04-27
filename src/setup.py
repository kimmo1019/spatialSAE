import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="spatialSAE", 
    version="1.0.0",
    author="Qiao Liu",
    author_email="liuqiao@stanford.edu",
    description="spatialSAE: Deciphering spatial transcritomics data with structural deep autoencoder networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kimmo1019/spatialSAE",
    packages=setuptools.find_packages(),
    install_requires=["tensorflow-gpu>=2.4.1","python-igraph","pandas","numpy","scipy","scanpy","anndata","louvain","sklearn", "numba", "re"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)