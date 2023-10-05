#!/bin/bash

#SBATCH --job-name dwld

#SBATCH -N 1

#SBATCH -n 1

#SBATCH --mem=4G

#SBATCH --mail-type=ALL
#SBATCH --nodelist=reslab32ai8111

#SBATCH --mail-user=go2432@wayne.edu

#SBATCH -o output_%j.out

#SBATCH -e errors_%j.err

#SBATCH -t 28-0:0:0
bash ./download_all.txt
