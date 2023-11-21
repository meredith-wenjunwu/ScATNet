import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import pdb

if __name__ == '__main__':
    # labels = ['7.5x', '10x', '12.5x', '7.5x + 10x', '7.5x + 12.5x', '10x + 12.5x', '7.5x + 10x + 12.5x']
    labels = ['ChikonMIL', 'MS-DA-MIL (MSC)', 'Ours (7.5x + 10x)']
    fig = plt.figure()
    colors = cm.get_cmap('tab10').colors
    acc_array = np.array([
        [0.6667, 0.3348, 0.3548, 0.9583],
        [0.6207, 0.3667, 0.5882, 0.8181],
        [0.7931, 0.4000, 0.6333, 0.7727]
    ])
    # acc_array = np.array(
    #     [[0.7241, 0.4333, 0.5667,0.4545],
    #     [0.8276, 0.4667, 0.6667, 0.4091],
    #     [0.6897, 0.6000, 0.5667, 0.5909],
    #     [0.7931, 0.4000, 0.6333, 0.7727],
    #     [0.6207, 0.6000, 0.5000, 0.8636],
    #     [0.6552, 0.5333, 0.6333, 0.7273 ],
    #     [0.6897, 0.5667, 0.6333, 0.6364]])

    xaxis = ['MMD', 'MIS','pT1a', 'pT1b']
    width = 0.25
    plt.rcParams.update({'font.size': 18})
    space = 0.01
    ind = np.arange(0, len(xaxis))
    # pdb.set_trace()
    for i, l in enumerate(labels):
        data = list(acc_array[i, :])
        if i <= 2:
            color = colors[i]
        else:
            color = colors[i+1]
        # else:
            # color=colors[4]
        plt.bar(ind + i * width, data, width*0.9, label=l, color=color)

    plt.ylabel('Accuracy')
    plt.xticks(ind + width/2, xaxis)
    plt.xlabel('Diagnostic Category')
    ax = plt.gca()
    ax.set_ylim([0, 1.0])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.legend(loc='upper left', ncol=1, labelspacing=0.05)
    fig.savefig('ours_baseline_roc.pdf', dpi=300, bbox_inches='tight')



