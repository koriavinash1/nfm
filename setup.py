import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
     name='nfm',  
     version='0.0.1',
     author="Avinash Kori",
     author_email="koriavinash1@gmail.com",
     description="Neural Field Models",
     long_description=open("README.md").read(),
   long_description_content_type="text/markdown",
     url="https://github.com/koriavinash1/nfm",
     packages=setuptools.find_packages(),
     install_requires = [
         'tqdm',
         'numpy',
         'pandas',
         'matplotlib',
         'scipy'
         ],
     classifiers=[
         "Programming Language :: Python :: 3.6",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )
