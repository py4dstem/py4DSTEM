from .vis_grid import show_image_grid

def _show_grid(**kwargs):
    """
    """
    assert 'ar' in kwargs.keys()
    ar = kwargs['ar']
    del kwargs['ar']

    # parse grid of images
    if isinstance(ar[0],list):
        assert(all([isinstance(ar[i],list) for i in range(len(ar))]))
        W = len(ar[0])
        H = len(ar)
        def get_ar(i):
            h = i//W
            w = i%W
            try:
                return ar[h][w]
            except IndexError:
                return
    else:
        W = len(ar)
        H = 1
        def get_ar(i):
            return ar[i]

    if kwargs['returnfig']:
        return show_image_grid(get_ar,H,W,**kwargs)
    else:
        show_image_grid(get_ar,H,W,**kwargs)



