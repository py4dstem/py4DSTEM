# `py4DSTEM.visualize`

Visualization functions.  The basic visualization function has a call signature

```
show(ar,min=0,max=3,power=1,figsize=(12,12),contrast='std',ax=None,
     bordercolor=None,borderwidth=5,returnfig=False,cmap='gray',**kwargs)
```

Most other visualization functions are built on top of this one, and accept these arguments, possibly plus others.  Additional keyword arguments passed as `**kwargs` are passed to `plt.show`.  Creating and then performing additional edits to a plot is accomplished by setting `returnfig=True`, which then returns a 2-tuple `(fig,ax)`.

