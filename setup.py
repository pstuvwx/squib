import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="squib",
    version="0.0.1",
    author="pstuvwx",
    author_email="ksrgmiyabi@gmail.com",
    description="A library for blowing up your PyTorch project.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pstuvwx/squib",
    packages=['squib'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)