#!/bin/bash
source ~/.bashrc
reset

# enter environment
env_name="pytorch"
# env_name=${PWD##*/}
conda activate $env_name
conda env list
python --version


# export
rm requirements.txt
pip list --format=freeze > requirements.txt
rm environment.yml
conda env export > environment.yml