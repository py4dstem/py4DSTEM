import py4DSTEM

fp_v050 = "/Users/Ben/Desktop/sample_v050.h5"
fp_v070 = "/Users/Ben/Desktop/sample_v070.h5"
fp2_v070 = "/Users/Ben/Desktop/sample2_v070.h5"


#print('testing on v050...')
#py4DSTEM.io.read(fp_v050)
#data = py4DSTEM.io.read(fp_v050,data_id='ppotential')
#print(data)
#print('All good!')



print('testing on v070...')
#py4DSTEM.io.read(fp_v070)
pla = py4DSTEM.io.read(fp_v070,data_id='pointlistarray_0')
print(pla)
#py4DSTEM.io.read(fp2_v070)
#dc = py4DSTEM.io.read(fp2_v070,data_id='datacube_0')
#print(dc)
print('yayayayayayayay')


