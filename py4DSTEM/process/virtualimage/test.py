# Tests for virtual image module

if __name__=="__main__":

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle, Wedge
    import py4DSTEM
    from py4DSTEM.process.virtualimage import get_virtualimage_rect, get_virtualimage_circ, get_virtualimage_ann

    # Load test data
    fp = "testdata_10x10_ss=100_alpha=p48_spot=11_cl=1200_kV=300_RT_bin=4_DT=0p5s.dm3"
    dc = py4DSTEM.file.io.read(fp)
    dc.set_scan_shape(10,10)

    # Set params
    dp_x,dp_y = 5,8               # Sample DP
    xmin,xmax = 206,306           # Rectangular detector shape
    ymin,ymax = 226,286
    x0,y0 = 240,270               # Center for circular and annular detectors
    R = 30                        # Circular detector radius
    Ri,Ro = 80,120                # Annular detector radii

    # Get images
    dp = dc.data[dp_x,dp_y,:,:]
    virtual_image_rect = get_virtualimage_rect(dc,xmin,xmax,ymin,ymax)
    virtual_image_circ = get_virtualimage_circ(dc,x0,y0,R)
    virtual_image_ann = get_virtualimage_ann(dc,x0,y0,Ri,Ro)

    # Patches for visualization of detectors
    rect_detector = Rectangle((ymin-0.5,xmin-0.5),ymax-ymin,xmax-xmin,color='r',alpha=0.3)
    circ_detector = Circle((y0-0.5,x0-0.5),R,color='r',alpha=0.3)
    ann_detector = Wedge((y0-0.5,x0-0.5),Ro,0,360,width=Ro-Ri,color='r',alpha=0.3)

    # Plot
    fig,axs = plt.subplots(3,2,figsize=(6,9))
    axs[0,0].matshow(dp)
    axs[1,0].matshow(dp)
    axs[2,0].matshow(dp)
    axs[0,0].add_patch(rect_detector)
    axs[1,0].add_patch(circ_detector)
    axs[2,0].add_patch(ann_detector)
    axs[0,1].matshow(virtual_image_rect)
    axs[1,1].matshow(virtual_image_circ)
    axs[2,1].matshow(virtual_image_ann)
    axs[0,1].scatter(dp_y,dp_x,color='r')
    axs[1,1].scatter(dp_y,dp_x,color='r')
    axs[2,1].scatter(dp_y,dp_x,color='r')
    plt.show()


