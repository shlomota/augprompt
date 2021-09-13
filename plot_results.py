import matplotlib.pyplot as plt

SMALL_SIZE = 10
SMALLISH_SIZE = 12
MEDIUM_SIZE = 14
BIG_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALLISH_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALLISH_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIG_SIZE)  # fontsize of the figure title


n_vals = [10, 20, 50, 100]
# n_vals = [10, 25, 50, 100]
m_vals = [0, 0.25, 0.5, 1, 2, 5]
m_vals = [x*2 for x in m_vals] # my values need to be multiplied
m_vals = [0, 0.25, 0.5, 1, 2]
results = [
           [0.499, 0.501, 0.622, 0.523, 0.546, 0.635],
           [0.519, 0.554, 0.616, 0.618, 0.618, 0.651],
           [0.706, 0.620, 0.654, 0.699, 0.701, 0.762],
           [0.805, 0.814, 0.762, 0.760, 0.769, 0.708]
]

#quora
#b4fix
results = [[0.6278, 0.6314, 0.6278, 0.6272, 0.6262],
           [0.633, 0.6304, 0.6288, 0.6242, 0.635],
           [0.6418, 0.6294, 0.6364, 0.6418, 0.6312],
           [0.679, 0.6648, 0.6778, 0.65, 0.6378]]
#i_1
results = [[0.62, 0.623, 0.62, 0.62, 0.623],
           [0.62, 0.62, 0.62, 0.64, 0.621],
           [0.646, 0.633, 0.642, 0.631, 0.634],
           [0.667, 0.672, 0.674, 0.66, 0.682]]

#paranmt large except 50
# results_old = [[0.3084, 0.3174, 0.2944, 0.2806, 0.2582],
#            [0.4074, 0.3648, 0.3676, 0.4112, 0.3722],
#            [0.7796, 0.785, 0.8028, 0.8046, 0.772],
#            [0.9778, 0.9658, 0.9708, 0.9562, 0.9304]]

#paranmt xl
# results_old = [[0.3084, 0.2784, 0.2692, 0.3036, 0.2728],
#  [0.4074, 0.4166, 0.3806, 0.361, 0.3818],
#  [0.7796, 0.785, 0.8028, 0.8046, 0.772],
#  [0.9778, 0.973, 0.9584, 0.9412, 0.8988]]

results = [[0.286, 0.345, 0.349, 0.286, 0.302],
           [0.305, 0.32, 0.277, 0.277, 0.319],
           [0.877, 0.925, 0.664, 0.813, 0.734],
           [0.979, 0.965, 0.961, 0.964, 0.943]]

for i in range(len(n_vals)):
  plt.plot(m_vals, results[i], marker="o", label="n = " + str(n_vals[i]))
  # plt.plot(m_vals, results[i], label="n = " + str(n_vals[i]))

# plt.legend(loc="top right")
plt.legend(loc="bottom right")
# plt.legend(loc="bottom right")
# plt.legend(bbox_to_anchor=(1.2, 0.5), loc="lower right")
#plt.legend()
# naming the x axis
# plt.xlabel('Dataset Size')
plt.xlabel('Augmentation Multiplicity')
# naming the y axis
plt.ylabel('Accuracy')

# giving a title to my graph
# plt.title('Accuracy Results on the PARANMT Dataset')
# plt.title('Accuracy Results on the Quora Questions Pairs Dataset')
# plt.title('Model accuracy on the Quora Questions Pairs Dataset')
plt.title('Model accuracy on the PARANMT Dataset')

# function to show the plot
plt.show()