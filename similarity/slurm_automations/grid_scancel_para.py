# !python train_quora.py -m 0.5 -n 10 -i 5
import os
a = "!python train_quora.py -m %s -n %d -i 5"
a = "python train_quora.py -m %s -n %d -i 5"
ns = [10, 25, 50, 100]
ms = [0, 0.25, 0.5, 1, 2]




for n in ns:
    for m in ms:
        print(a % (m, n))
        s = "para_%d_%s" % (n,m)
        os.system("scancel -n %s" % (s))



