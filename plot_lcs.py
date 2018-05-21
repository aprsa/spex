import numpy as np
import matplotlib.pyplot as plt
import os

for f in os.listdir('lcs/'):
    test = np.genfromtxt('lcs/%s' % f, skip_header=21, skip_footer=1, usecols=np.arange(1,8)).T
    plt.xlabel('Time')
    plt.ylabel('Magnitude')
    plt.gca().invert_yaxis()
    plt.plot(test[0], test[1], 'b-', label='u')
    plt.plot(test[0], test[2], 'g-', label='g')
    plt.plot(test[0], test[3], 'y-', label='r')
    plt.plot(test[0], test[4], 'r-', label='i')
    plt.plot(test[0], test[5], 'm-', label='z')
    plt.plot(test[0], test[6], 'k-', label='y')
    plt.legend(loc='lower right')
    plt.show()
