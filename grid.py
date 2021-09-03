# !python train_quora.py -m 0.5 -n 10 -i 5
a = "!python train_quora.py -m %.1f -n %d -i 5"
ns = [10, 100, 500]
ms = [0, 0.5, 1, 2, 5]

for n in ns:
    for m in ms:
        print(a % (m, n))