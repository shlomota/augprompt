# !python train_quora.py -m 0.5 -n 10 -i 5
import os
a = "!python train_quora.py -m %.1f -n %d -i 5"
a = "python train_quora.py -m %.1f -n %d -i 5"
ns = [10, 25, 50, 100]
ms = [0, 0.25, 0.5, 1, 2]


full_path = False

template = """#! /bin/sh
#SBATCH --output=/home/yandex/AMNLP2021/shlomotannor/amnlp/shlomo/quora_out/%s/out.out
#SBATCH --error=/home/yandex/AMNLP2021/shlomotannor/amnlp/shlomo/quora_out/%s/err.err
#SBATCH --partition=studentkillable
#SBATCH --job-name=%s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
/home/yandex/AMNLP2021/shlomotannor/anaconda3/envs/nli/bin/python /home/yandex/AMNLP2021/shlomotannor/amnlp/shlomo/augprompt/train_quora.py -i 5 -n %d -m %.1f
"""

template_int = """#! /bin/sh
#SBATCH --output=/home/yandex/AMNLP2021/shlomotannor/amnlp/shlomo/quora_out/%s/out.out
#SBATCH --error=/home/yandex/AMNLP2021/shlomotannor/amnlp/shlomo/quora_out/%s/err.err
#SBATCH --partition=studentkillable
#SBATCH --job-name=%s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
/home/yandex/AMNLP2021/shlomotannor/anaconda3/envs/nli/bin/python /home/yandex/AMNLP2021/shlomotannor/amnlp/shlomo/augprompt/train_quora.py -i 5 -n %d -m %d --aug-only True
"""


if not full_path:
    template = """#! /bin/sh
#SBATCH --output=../quora_out/%s.out
#SBATCH --error=../quora_out/%s.err
#SBATCH --partition=studentkillable
#SBATCH --job-name=%s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
/home/yandex/AMNLP2021/shlomotannor/anaconda3/envs/nli/bin/python /home/yandex/AMNLP2021/shlomotannor/amnlp/shlomo/augprompt/train_quora.py -i 5 -n %d -m %.1f --aug-only True
"""

    template_int = """#! /bin/sh
#SBATCH --output=../quora_out/%s.out
#SBATCH --error=../quora_out/%s.err
#SBATCH --partition=studentkillable
#SBATCH --job-name=%s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
/home/yandex/AMNLP2021/shlomotannor/anaconda3/envs/nli/bin/python /home/yandex/AMNLP2021/shlomotannor/amnlp/shlomo/augprompt/train_quora.py -i 5 -n %d -m %d --aug-only True
"""


for n in ns:
    m = 2
    print(a % (m, n))
    s = "%d_%d" % (n,m)
    slurm_content = template_int % (s, s, s, n, m)
    with open("/home/yandex/AMNLP2021/shlomotannor/amnlp/shlomo/augprompt/quora_%s.slurm" % (s), "w") as f:
        f.write(slurm_content)
    os.system("sbatch /home/yandex/AMNLP2021/shlomotannor/amnlp/shlomo/augprompt/quora_%s.slurm" % (s))



