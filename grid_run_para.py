# !python train_paranmt.py -m 0.5 -n 10 -i 5
import os
import time
time.sleep(60 * 60 * 3)
a = "!python train_paranmt.py -m %s -n %d -i 5"
a = "python train_paranmt.py -m %s -n %d -i 5"
ns = [10, 25, 50, 100]
ns = [10, 25, 100]
# ns = [10, 100]
ms = [0, 0.25, 0.5, 1, 2]


full_path = False

template = """#! /bin/sh
#SBATCH --output=/home/yandex/AMNLP2021/shlomotannor/amnlp/shlomo/paranmt_out/%s/out.out
#SBATCH --error=/home/yandex/AMNLP2021/shlomotannor/amnlp/shlomo/paranmt_out/%s/err.err
#SBATCH --partition=studentkillable
#SBATCH --job-name=%s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
/home/yandex/AMNLP2021/shlomotannor/anaconda3/envs/nli/bin/python /home/yandex/AMNLP2021/shlomotannor/amnlp/shlomo/augprompt/train_paranmt.py -i 5 -n %d -m %s
"""

if not full_path:
    template = """#! /bin/sh
#SBATCH --output=../paranmt_out/%s.out
#SBATCH --error=../paranmt_out/%s.err
#SBATCH --partition=studentkillable
#SBATCH --job-name=%s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
/home/yandex/AMNLP2021/shlomotannor/anaconda3/envs/nli/bin/python /home/yandex/AMNLP2021/shlomotannor/amnlp/shlomo/augprompt/train_paranmt.py -i 5 -n %d -m %s
"""


for n in ns:
    for m in ms:
        print(a % (m, n))
        s = "para_%d_%s" % (n,m)
        slurm_content = template % (s, s, s, n, m)

        with open("/home/yandex/AMNLP2021/shlomotannor/amnlp/shlomo/augprompt/paranmt_%s.slurm" % (s), "w") as f:
            f.write(slurm_content)
        os.system("sbatch /home/yandex/AMNLP2021/shlomotannor/amnlp/shlomo/augprompt/paranmt_%s.slurm" % (s))



