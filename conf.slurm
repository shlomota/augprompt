#! /bin/sh
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err
#SBATCH --partition=studentkillable
#SBATCH --job-name daprompt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
/home/yandex/AMNLP2021/shlomotannor/anaconda3/envs/amnlp/bin/python train_sst.py -t b -m 5 -e 10 -n 100 -s True -i 5
