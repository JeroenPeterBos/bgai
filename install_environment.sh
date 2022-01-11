#!/bin/zsh

CONDA_ENVIRONMENT=reinforcement_learning
echo $CONDA_PREFIX

if [[ $CONDA_PREFIX == *$CONDA_ENVIRONMENT ]]; then
  echo "\nPlease first run 'conda deactivate' to deactivate the $CONDA_ENVIRONMENT conda environment!\n" 
  exit
fi

conda remove -n $CONDA_ENVIRONMENT --all -y
conda env create -n $CONDA_ENVIRONMENT -f conda.yml