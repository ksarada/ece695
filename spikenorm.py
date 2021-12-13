import numpy as np
import matplotlib.pyplot as plt
import torch


#a = np.load('here.npz')
act = np.load('here_plot_act.npz')
val = np.load('here_plot_val.npz')

val_mean = np.mean(val['arr_0'], axis=0)
act_mean = np.mean(act['arr_0'], axis=0)

plt.plot(act_mean.flatten()[0:100], val_mean.flatten()[0:100])
plt.show()
