import numpy as np
import matplotlib.pyplot as plt

def show_kernel(kernel,R,L,W,figsize=(12,6)):
    """
    Plots, side by side, the probe kernel and its line profile.
    R is the kernel plot's window size.
    L and W are the lenth ad width of the lineprofile.
    """
    lineprofile = np.concatenate([np.sum(kernel[-L:,:W],axis=(1)),
                                  np.sum(kernel[:L,:W],axis=(1))])

    fig,(ax1,ax2) = plt.subplots(1,2,figsize=figsize)
    ax1.matshow(kernel[:int(R),:int(R)],cmap='gray')
    ax2.plot(np.arange(len(lineprofile)),lineprofile)
    plt.show()
    return


