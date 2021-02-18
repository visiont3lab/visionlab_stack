import setuptools

# https://packaging.python.org/guides/distributing-packages-using-setuptools/
# python3 setup.py sdist bdist_wheel

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    
setuptools.setup(
    name="visionlab_stack",
    version="0.0.1",
    author="Manuel Rucci",
    author_email="ruccimanuel7@gmail.com",
    description="A collection of computer vision and deep learning algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/visiont3lab/visionlab_stack",
    #include_package_data=True,
    packages=setuptools.find_packages(),
    install_requires=[
        'Cython',
        'Pillow',
        'PyYAML>=5.3.1',
        'opencv-python==4.2.0.34',
        'torch>=1.7.0',
        'torchvision>=0.8.1',
        'tqdm>=4.41.0',
        'neuralgym @ git+https://github.com/JiahuiYu/neuralgym',
        'tensorflow==1.14.0',
        'imutils',
        'requests',
        'matplotlib',
        'pandas',
        'seaborn'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU GENERAL PUBLIC License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
