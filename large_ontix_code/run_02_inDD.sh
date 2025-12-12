#!/bin/bash

# module loads
module purge
module load release/25.06

# Retrieve chunks in parallel 
for i in {0..12};do # Parallelize on HPC for speed up (takes around 24h)
	sbatch <<-EOT
	#!/bin/bash
	#SBATCH --job-name=large_ontix_census
	#SBATCH --output=./logs/slurm_%a_%j.out
	#SBATCH --error=./logs/slurm_%a_%j.err
	#SBATCH --nodes=1
	#SBATCH --tasks-per-node=1
	#SBATCH --cpus-per-task=8
	#SBATCH --mem-per-cpu=4096
	#SBATCH --time=1:00:00

	source .venv/bin/activate
	# Define chunks
	python large_ontix_code/01_get_census_data.py "step2" $i "test" # Test mode with only 48 chunks for quick testing

	exit 0
	EOT
done
