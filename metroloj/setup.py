import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='metroloj',  
     version='0.1.2',
     scripts=['metroloj_script'] ,
     author="Youssef Ayachi",
     author_email="youssef-ayachi@outlook.com",
     description="A python implementation of ImageJ MetroloJ plugin",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/javatechy/dokr",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )