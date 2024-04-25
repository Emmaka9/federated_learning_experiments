#!/bin/bash
#SBATCH --partition comp06
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --job-name="output_flwr_cifar10_paillier"
#SBATCH --output=%x-%j.out

cd /home/ikemmaka/workplace_emmamka/fl_experiments/federated_learning_experiments/flower_with_paillier_enc
module load python
source activate federated_learning
python flwr_cifar10_paillier.py

