from setuptools import setup, find_packages


setup(
    name="tdb",
    version="0.2.0",
    author="Konrad Szafer",
    description="",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="",

    python_requires='>=3.9',
    install_requires=[
        'torch>=1.12.0'
    ],

    package_dir={'': '.'},
    packages=find_packages(
        where='.',
        exclude=['tests']
    ),
    
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
    ],
)