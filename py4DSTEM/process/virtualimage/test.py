# Tests for virtual image module

import py4DSTEM
import matplotlib.pyplot as plt


if __name__=="__main__":

    from py4DSTEM.process.virtualimage import get_virtualimage_rect

    fp = "testdata_10x10_ss=100_alpha=p48_spot=11_cl=1200_kV=300_RT_bin=4_DT=0p5s.dm3"
    dc = py4DSTEM.file.io.read(fp)
    dc.set_scan_shape(10,10)

    virtual_image = get_virtualimage_rect(dc,xmin=206,xmax=306,ymin=206,ymax=306)

    fig,ax = plt.subplots()
    ax.matshow(virtual_image)
    plt.show()


