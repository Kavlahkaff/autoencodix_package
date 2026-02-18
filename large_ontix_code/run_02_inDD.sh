#!/bin/bash

# module loads
module purge
module load release/25.06

# Retrieve chunks in parallel 
total_parts=64
select_parts_list=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16) # Rerun only specific parts that failed
# for i in $(seq 0 $total_parts);do # Parallelize on HPC for speed up (takes around 24h)
for i in "${select_parts_list[@]}";do # Rerun specific parts only
	echo "Submitting job for part $i out of $total_parts"
	sbatch <<-EOT
	#!/bin/bash
	#SBATCH --job-name=large_ontix_census
	#SBATCH --output=./logs/slurm_%a_%j.out
	#SBATCH --error=./logs/slurm_%a_%j.err
	#SBATCH --nodes=1
	#SBATCH --tasks-per-node=1
	#SBATCH --cpus-per-task=16
	#SBATCH --mem-per-cpu=48000
	#SBATCH --time=48:00:00
	#SBATCH --account=p_scads_autoencodix


	source .venv/bin/activate
	# Define chunks
	python large_ontix_code/01_get_census_data.py "step2" $i "full" $total_parts # Test mode with only 48 chunks for quick testing

	exit 0
	EOT
done
