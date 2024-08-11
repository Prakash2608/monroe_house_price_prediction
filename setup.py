from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT = '-e .'

"""HYPEN_E_DOT='-e .'

def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements"""
    
    
__version__ = "0.0.1"
REPO_NAME = "monroe_house_price_prediction"
PKG_NAME= "monroe_hpp_test"
AUTHOR_USER_NAME = "Prakash2608"
AUTHOR_EMAIL = "prakashraj822@gmail.com"

setup(
    name=REPO_NAME,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    # install_requires=["scikit-learn","pandas","numpy"],
    packages=find_packages()
)