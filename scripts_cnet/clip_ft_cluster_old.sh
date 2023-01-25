
#!/bin/bash

#SBATCH --job-name=clip_ft
#SBATCH --nodes=1        
#SBATCH --ntasks-per-node=1
#SBATCH --partition=mid
#SBATCH --time=1-0:0:0        
#SBATCH --output=logs/clip_ft-%j.out
#SBATCH --mem=32G
# Load Anaconda
echo "======================="
echo "Loading Anaconda Module..."
#module load anaconda/2.7
module load cuda/11.4
module load cudnn/8.2.2/cuda-11.4 
export HYDRA_FULL_ERROR=1

python -u clip_ft_pred.py