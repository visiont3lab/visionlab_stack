import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="visionlab_stack",
    version="0.0.1",
    author="Manuel Rucci",
    author_email="ruccimanuel7@gmail.com",
    description="A collection of fast detectors to use in real time scenario",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU GENERAL PUBLIC License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
