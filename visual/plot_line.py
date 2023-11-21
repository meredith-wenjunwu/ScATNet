from matplotlib import pyplot as plt
import numpy as np
from matplotlib import cm

def convert(arr):
    return [round(v/100, 2) for v in arr]
def main():
    n_75x = convert([54.95, 54.05, 47.75, 52.25, 47.75, 49.55])
    n_100x = convert([58.56, 60.36, 50.45, 45.05, 50.45, 46.85])
    n_125x = convert([59.46, 57.66, 61.26, 57.66, 43.24, 48.65])
    x_axis = [i**2 for i in range(5, 17, 2)]
    print(x_axis)
    ax = plt.subplot(111)
    colors = cm.get_cmap('tab20c').colors
    colors2 = cm.get_cmap('tab20b').colors
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.plot(n_75x, label='7.5x', marker="o", markersize="10", linestyle='dashed', color='tab:blue')
    ax.plot(n_100x, label='10x', marker="o", markersize="10", linestyle='dashed', color='tab:orange')
    ax.plot(n_125x, label='12.5x', marker="o", markersize="10", linestyle='dashed', color='tab:green')
    ax.legend(fontsize=16, edgecolor='k', fancybox=True)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ind = np.arange(len(x_axis))
    ax.set_xticks(ind)
    ax.set_xticklabels(x_axis)
    plt.xlabel('Number of crops (m)', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.savefig('n_crop.pdf', bbox_inches='tight',pad_inches=0)
if __name__ == "__main__":
    main()