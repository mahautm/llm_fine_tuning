#!/bin/bash

# Parameters
#SBATCH --mem=64G
##SBATCH --cpus-per-task=48
#SBATCH --partition=alien
#SBATCH --gres=gpu:1
#SBATCH --qos=alien
#SBATCH --exclude=node044
#SBATCH --error=/home/mmahaut/projects/experiments/dl/%j_0_log.err
#SBATCH --job-name=wikidata
#SBATCH --output=/home/mmahaut/projects/experiments/dl/%j_0_log.out
#source /etc/profile.d/zz_hpcnow-arch.sh
source ~/.bashrc
# source /home/mmahaut/.bashrc

echo $SLURMD_NODENAME
conda activate py39
export PATH=$PATH:/soft/easybuild/x86_64/software/Miniconda3/4.9.2/bin/
which python
export PATH=$PATH:~/projects/simple-wikidata-db/
python -m pip install ~/projects/simple-wikidata-db/
cd ~/projects/simple-wikidata-db/
wget https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.gz

# print file size
ls -lh latest-all.json.gz
# check file is complete. add eof to json if required
zcat latest-all.json.gz | tail -n 1 | grep -q '^\]$' || echo ']' >> latest-all.json
python ~/projects/simple-wikidata-db/simple_wikidata_db/preprocess_dump.py --input_file ~/projects/simple-wikidata-db/latest-all.json.gz --out_dir /projects/colt/processed-wikidata-28-2-2024 --num_lines_in_dump=95980335
cd ~/projects/parametric_mem/
python /home/mmahaut/projects/parametric_mem/fetch_triples_and_aliases.py --processed_wikidata=/projects/colt/processed-wikidata-28-2-2024/
echo "Extraction DONE"
python /home/mmahaut/projects/parametric_mem/filter_triples.py --processed_wikidata=/projects/colt/processed-wikidata-28-2-2024/
echo "Filtering DONE"
python /home/mmahaut/projects/parametric_mem/create_benchmark.py 
