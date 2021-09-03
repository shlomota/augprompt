#!python train_quora.py -t n -m 0.5 -e 5 -n 10 -f False

training_loss = [4.8424, 1.3431, 0.4293, 0.1975, 0.1042]
val_loss = [2.2997, 0.803, 0.846, 0.822, 0.805]
val_acc = [0.66, 0.65, 0.64, 0.49, 0.55, 0.56]
import matplotlib.pyplot as plt
plt.plot(range(1,6), training_loss, label="training_loss")
plt.plot(range(1,6), val_loss, label="val_loss")
plt.legend()
plt.show()