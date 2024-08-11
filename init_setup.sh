echo [$(date)]: "START"


echo [$(date)]: "creating env with python 3.10 version" 


conda create --prefix ./monroe_hpp_env python=3.10 -y


echo [$(date)]: "activating the environment" 

conda activate ./monroe_hpp_env

echo [$(date)]: "installing the dev requirements" 

pip install -r requirements_dev.txt

echo [$(date)]: "END" 