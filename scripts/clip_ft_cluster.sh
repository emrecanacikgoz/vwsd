#!/bin/bash

#SBATCH --job-name=clip_ft
#SBATCH --nodes=1        
#SBATCH --ntasks-per-node=5
#SBATCH --partition=ai 
#SBATCH --qos=ai     
#SBATCH --account=ai   
#SBATCH --gres=gpu:tesla_t4:1    
#SBATCH --time=1-0:0:0        
#SBATCH --output=logs/clip_ft-%j.out
#SBATCH --mem=60G
# Load Anaconda
echo "======================="
echo "Loading Anaconda Module..."
#module load anaconda/2.7
module load cuda/11.4
module load cudnn/8.2.2/cuda-11.4 

python -u clip_ft_pred.py