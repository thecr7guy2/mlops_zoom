from setuptools import find_packages,setup

# def get_requirements(path):
#     with open(path) as f :
#         requirements = f.readlines()
#         a = [i.replace("\n","") for i in requirements]
#         if "-e ." in a:
#             a.remove("-e ." )
#     return a 


setup(
name = "Used_Car_Price_Detection",
version= "0.0.1",
author="Sai",
author_email="manirajadapa@gmail.com",
description='A small example machine learning project',
url='https://github.com/thecr7guy2/mlops_zoom',
packages=find_packages(),
# install_requires = get_requirements("requirements.txt")
)