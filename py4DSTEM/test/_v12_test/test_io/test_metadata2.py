import py4DSTEM

fp_dm = "/Users/Ben/Work/Data/py4DSTEM_sampleData/smallDataSets/small4DSTEMscan_10x10.dm3"
fp_h5 = "/Users/Ben/Work/Data/py4DSTEM_sampleData/smallDataSets/small4DSTEMscan_10x10.h5"
dc = py4DSTEM.io.read(fp_dm)
dc.name='dc'
dc.metadata.set_Q_pixel_size(100)
py4DSTEM.io.save(fp_h5,dc,overwrite=True)
md = py4DSTEM.io.read(fp_h5,metadata=True)
print(md.get_Q_pixel_size())


