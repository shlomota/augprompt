# !python train_quora.py -m 0.5 -n 10 -i 5
import os
a = "!python train_quora.py -m %.1f -n %d -i 5"
a = "python train_quora.py -m %.1f -n %d -i 5"
ns = [10, 25, 50, 100]
ms = [0, 0.25, 0.5, 1, 2]

template = """
#! /bin/sh
#SBATCH --output=quora_out/%s/out.out
#SBATCH --error=quora_out/%s/err.err
#SBATCH --partition=studentkillable
#SBATCH --job-name
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
/home/yandex/AMNLP2021/shlomotannor/anaconda3/envs/nli/bin/python /home/yandex/AMNLP2021/shlomotannor/amnlp/shlomo/augprompt/train_quora.py -i 5 -n %d -m %.1f
"""

for n in ns:
    for m in ms:
        print(a % (m, n))
        s = "%d_%.1f" % (n,m)
        slurm_content = template % (s, s, n, m)
        with open("quora_%s.slurm" % (s), "w") as f:
            f.write(slurm_content)
        os.system("quora_%s.slurm" % (s))



