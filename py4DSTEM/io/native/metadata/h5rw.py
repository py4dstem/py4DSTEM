import h5py
import numpy as np
# import cPickle
import time
import os.path
import glob

__all__ = ['h5write', 'h5append', 'h5read', 'h5info', 'h5options']

h5options = dict(
    H5RW_VERSION='0.1',
    H5PY_VERSION=h5py.version.version,
    # UNSUPPORTED = 'pickle'
    # UNSUPPORTED = 'ignore'
    UNSUPPORTED='fail',
    SLASH_ESCAPE='_SLASH_')


def sdebug(f):
    """
    debugging decorator for _store functions
    """

    def newf(*args, **kwds):
        print('{0:20} {1:20}'.format(f.func_name, args[2]))
        return f(*args, **kwds)

    newf.__doc__ = f.__doc__
    return newf


def _h5write(filename, mode, grouppath, *args, **kwargs):
    """\
    _h5write(filename, mode, grouppath, {'var1'=..., 'var2'=..., ...})
    _h5write(filename, mode, grouppath, var1=..., var2=..., ...)
    _h5write(filename, mode, grouppath, dict, var1=..., var2=...)

    Writes variables var1, var2, ... to file filename, into a group
    located within the h5 file at grouppath, which should be a string
    with the format "outergroup/anothergroup/innergroup". The file mode
    can be chosen according to the h5py documentation. The key-value
    arguments have precedence on the provided dictionnary.

    supported variable types are:
    * scalars
    * numpy arrays
    * strings
    * lists
    * dictionaries

    (if the option UNSUPPORTED is equal to 'pickle', any other type
    is pickled and saved. UNSUPPORTED = 'ignore' silently eliminates
    unsupported types. Default is 'fail', which raises an error.)

    The file mode can be chosen according to the h5py documentation.
    It defaults to overwriting an existing file.
    """

    filename = os.path.abspath(os.path.expanduser(filename))

    ctime = time.asctime()
    mtime = ctime

    # Update input dictionnary
    if args:
        d = args[0].copy()  # shallow copy
    else:
        d = {}
    d.update(kwargs)

    # List of object ids to make sure we are not saving something twice.
    ids = []

    # This is needed to store strings
    dt = h5py.special_dtype(vlen=str)

    def check_id(id):
        if id in ids:
            raise RuntimeError('Circular reference detected! Aborting save.')
        else:
            ids.append(id)

    def pop_id(id):
        ids[:] = [x for x in ids if x != id]

    # @sdebug
    def _store_numpy(group, a, name, compress=True):
        if compress:
            dset = group.create_dataset(name, data=a, compression='gzip')
        else:
            dset = group.create_dataset(name, data=a)
        dset.attrs['type'] = 'array'
        return dset

    # @sdebug
    def _store_string(group, s, name):
        dset = group.create_dataset(name, data=np.string_(s), dtype=dt)
        dset.attrs['type'] = 'string'
        return dset

    # @sdebug
    def _store_unicode(group, s, name):
        dset = group.create_dataset(name, data=np.asarray(s.encode('utf8')), dtype=dt)
        dset.attrs['type'] = 'unicode'
        return dset

    # @sdebug
    def _store_list(group, l, name):
        check_id(id(l))
        arrayOK = len(set([type(x) for x in l])) == 1
        if arrayOK:
            try:
                # Try conversion to a numpy array
                la = np.array(l)
                if la.dtype.type is np.string_:
                    arrayOK = False
                else:
                    dset = _store_numpy(group, la, name)
                    dset.attrs['type'] = 'arraylist'
            except:
                arrayOK = False
        if not arrayOK:
            # inhomogenous list. Store all elements individually
            dset = group.create_group(name)
            for i, v in enumerate(l):
                _store(dset, v, '%05d' % i)
            dset.attrs['type'] = 'list'
        pop_id(id(l))
        return dset

    # @sdebug
    def _store_tuple(group, t, name):
        dset = _store_list(group, list(t), name)
        dset_type = dset.attrs['type']
        dset.attrs['type'] = 'arraytuple' if dset_type == 'arraylist' else 'tuple'
        return dset

    # @sdebug
    def _store_dict(group, d, name):
        check_id(id(d))
        if any([type(k) not in [str] for k in d.keys()]):
            raise RuntimeError('Only dictionaries with string keys are supported.')
        dset = group.create_group(name)
        dset.attrs['type'] = 'dict'
        for k, v in d.items():
            if k.find('/') > -1:
                k = k.replace('/', h5options['SLASH_ESCAPE'])
                ndset = _store(dset, v, k)
                if ndset is not None:
                    ndset.attrs['escaped'] = '1'
            else:
                _store(dset, v, k)
        pop_id(id(d))
        return dset

    def _store_dict_new(group, d, name):
        check_id(id(d))
        dset = group.create_group(name)
        dset.attrs['type'] = 'dict'
        for i, kv in enumerate(d.items()):
            _store(dset, kv, '%05d' % i)
        pop_id(id(d))
        return dset

    # @sdebug
    def _store_None(group, a, name):
        dset = group.create_dataset(name, data=np.zeros((1,)))
        dset.attrs['type'] = 'None'
        return dset

    # @sdebug
    # def _store_pickle(group, a, name):
    #     apic = cPickle.dumps(a)
    #     dset = group.create_dataset(name, data=np.asarray(apic), dtype=dt)
    #     dset.attrs['type'] = 'pickle'
    #     return dset

    # @sdebug
    def _store(group, a, name):
        if name in group.keys():
            del group[name]
        if type(a) is str:
            dset = _store_string(group, a, name)
        # elif type(a) is unicode:
        #     dset = _store_unicode(group, a, name)
        elif type(a) is dict:
            dset = _store_dict(group, a, name)
        elif type(a) is list:
            dset = _store_list(group, a, name)
        elif type(a) is tuple:
            dset = _store_tuple(group, a, name)
        elif type(a) is np.ndarray:
            dset = _store_numpy(group, a, name)
        elif np.isscalar(a):
            dset = _store_numpy(group, np.asarray(a), name, compress=False)
            dset.attrs['type'] = 'scalar'
        elif a is None:
            dset = _store_None(group, a, name)
        else:
            if h5options['UNSUPPORTED'] == 'fail':
                raise RuntimeError('Unsupported data type : %s' % type(a))
            elif h5options['UNSUPPORTED'] == 'pickle':
                dset = _store_pickle(group, a, name)
            else:
                dset = None
        return dset

    # Open the file and save everything
    with h5py.File(filename, mode) as f:
        try:
            group = f[grouppath]
        except KeyError:
            f.create_group(grouppath)
            group = f[grouppath]
        group.attrs['h5rw_version'] = h5options['H5RW_VERSION']
        group.attrs['ctime'] = ctime
        group.attrs['mtime'] = mtime
        for k, v in d.items():
            _store(group, v, k)

    return


def h5write(filename, grouppath, *args, **kwargs):
    """\
    h5write(filename, grouppath, {'var1'=..., 'var2'=..., ...})
    h5write(filename, grouppath, var1=..., var2=..., ...)
    h5write(filename, grouppath, dict, var1=..., var2=...)


    Writes variables var1, var2, ... to file filename, into a group
    located within the h5 file at grouppath, which should be a string
    with the format "outergroup/anothergroup/innergroup". The key-value
    arguments have precedence on the provided dictionary.

    supported variable types are:
    * scalars
    * numpy arrays
    * strings
    * lists
    * dictionaries

    (if the option UNSUPPORTED is equal to 'pickle', any other type
    is pickled and saved. UNSUPPORTED = 'ignore' silently eliminates
    unsupported types. Default is 'fail', which raises an error.)

    The file mode can be chosen according to the h5py documentation.
    It defaults to overwriting an existing file.
    """

    _h5write(filename, 'w', grouppath, *args, **kwargs)
    return


def h5append(filename, grouppath, *args, **kwargs):
    """\
    h5append(filename, grouppath, {'var1'=..., 'var2'=..., ...})
    h5append(filename, grouppath, var1=..., var2=..., ...)
    h5append(filename, grouppath, dict, var1=..., var2=...)

    Appends variables var1, var2, ... to file filename, into a group
    located within the h5 file at grouppath, which should be a string
    with the format "outergroup/anothergroup/innergroup". The key-value
    arguments have precedence on the provided dictionary.

    supported variable types are:
    * scalars
    * numpy arrays
    * strings
    * lists
    * dictionaries

    (if the option UNSUPPORTED is equal to 'pickle', any other type
    is pickled and saved. UNSUPPORTED = 'ignore' silently eliminates
    unsupported types. Default is 'fail', which raises an error.)

    The file mode can be chosen according to the h5py documentation.
    It defaults to overwriting an existing file.
    """

    _h5write(filename, 'a', grouppath, *args, **kwargs)
    return


def h5read(filename, grouppath, *args, **kwargs):
    """\
    h5read(filename, grouppath)
    h5read(filename, grouppath, s1, s2, ...)
    h5read(filename, grouppath, (s1,s2, ...))

    Read variables from a hdf5 file created with h5write and returns them as
    a dictionary.

    If specified, only variable named s1, s2, ... are loaded.

    Variable names support slicing and group access. For instance, provided
    that the file contains the appropriate objects, the following syntax is
    valid:

    a = h5read('file.h5', 'myarray[2:4]')
    a = h5read('file.h5', 'adict.thekeyIwant')

    h5read(filename_with_wildcard, ... , doglob=True)
    Reads sequentially all globbed filenames.

    """
    doglob = kwargs.get('doglob', None)

    # Used if we read a list of files
    fnames = []
    if not isinstance(filename, str):
        # We have a list
        fnames = filename
    else:
        if doglob is None:
            # glob only if there is a wildcard in the filename
            doglob = glob.has_magic(filename)
        if doglob:
            fnames = sorted(glob.glob(filename))
            if not fnames:
                raise IOError('%s : no match.' % filename)

    if fnames:
        # We are here only if globbing was allowed.
        dl = []
        # Loop over file names
        for f in fnames:
            # Call again, but this time without globbing.
            d = h5read(f, grouppath, *args, doglob=False, **kwargs)
            dl.append(d)
        return dl

    # We are here only if there was no globbing (fnames is empty)
    filename = os.path.abspath(os.path.expanduser(filename))

    def _load_dict_new(dset):
        d = {}
        keys = dset.keys()
        keys.sort()
        for k in keys:
            dk, dv = _load(dset[k])
            d[dk] = dv
        return d

    def _load_dict(dset):
        d = {}
        for k, v in dset.items():
            if v.attrs.get('escaped', None) is not None:
                k = k.replace(h5options['SLASH_ESCAPE'], '/')
            d[k] = _load(v)
        return d

    def _load_list(dset):
        l = []
        keys = dset.keys()
        keys.sort()
        for k in keys:
            l.append(_load(dset[k]))
        return l

    def _load_numpy(dset, sl=None):
        if sl is not None:
            return dset[sl]
        else:
            return dset[...]

    def _load_scalar(dset):
        try:
            return dset[...].item()
        except:
            return dset[...]

    def _load_str(dset):
        return dset[()]

    def _load_unicode(dset):
        return dset[()].decode('utf8')

    # def _load_pickle(dset):
    #     return cPickle.loads(dset[...])

    def _load(dset, sl=None):
        dset_type = dset.attrs.get('type', None)

        # Treat groups as dicts
        if (dset_type is None) and (type(dset) is h5py.Group):
            dset_type = 'dict'

        if dset_type == 'dict':
            if sl is not None:
                raise RuntimeError('Dictionaries do not support slicing')
            val = _load_dict(dset)
        elif dset_type == 'list':
            val = _load_list(dset)
            if sl is not None:
                val = val[sl]
        elif dset_type == 'array':
            val = _load_numpy(dset, sl)
        elif dset_type == 'arraylist':
            val = [x for x in _load_numpy(dset)]
            if sl is not None:
                val = val[sl]
        elif dset_type == 'tuple':
            val = tuple(_load_list(dset))
            if sl is not None:
                val = val[sl]
        elif dset_type == 'arraytuple':
            val = tuple(_load_numpy(dset).tolist())
            if sl is not None:
                val = val[sl]
        elif dset_type == 'string':
            val = _load_str(dset)
            if sl is not None:
                val = val[sl]
        elif dset_type == 'unicode':
            val = _load_str(dset)
            if sl is not None:
                val = val[sl]
        elif dset_type == 'scalar':
            val = _load_scalar(dset)
        elif dset_type == 'None':
            # 24.4.13 : B.E. commented due to hr5read not being able to return None type
            # try:
            #   val = _load_numpy(dset)
            # except:
            #    val = None
            val = None
        # elif dset_type == 'pickle':
        #     val = _load_pickle(dset)
        elif dset_type is None:
            val = _load_numpy(dset)
        else:
            raise RuntimeError('Unsupported data type : %s' % dset_type)
        return val

    outdict = {}
    with h5py.File(filename, 'r') as f:
        assert grouppath in f.keys(), "Specified group, {}, not found in this .h5 file.".format(grouppath)
        group = f[grouppath]
        if len(args) == 0:
            # no input arguments - load everything
            key_list = group.keys()
        else:
            if (len(args) == 1) and (type(args[0]) is list):
                # input argument is a list of object names
                key_list = args[0]
            else:
                # arguments form a list
                key_list = list(args)
        for k in key_list:
            # detect slicing
            if '[' in k:
                k, slice_string = k.split('[')
                slice_string = slice_string.split(']')[0]
                sldims = slice_string.split(',')
                sl = tuple(
                    [slice(*[None if x.strip() == '' else int(x) for x in (s.split(':') + ['', '', ''])[:3]]) for s in
                     sldims])
            else:
                sl = None

            # detect group access
            if '.' in k:
                glist = k.split('.')
                k = glist[-1]
                gr = group[glist[0]]
                for gname in glist[1:-1]:
                    gr = gr[gname]
                outdict[k] = _load(gr[k], sl)
            else:
                outdict[k] = _load(group[k], sl)

    return outdict


def h5info(filename, output=None):
    """\
    h5info(filename)

    Prints out a tree structure of given h5 file.

    [17/01/2012 guillaume potdevin]
    added optional argument output:
    if output is set to 1, then the printed string is returned
    """

    indent = 4
    filename = os.path.abspath(os.path.expanduser(filename))

    def _format_dict(key, dset):
        stringout = ' ' * key[0] + ' * %s [dict]:\n' % key[1]
        for k, v in dset.items():
            if v.attrs.get('escaped', None) is not None:
                k = k.replace(h5options['SLASH_ESCAPE'], '/')
            stringout += _format((key[0] + indent, k), v)
        return stringout

    def _format_list(key, dset):
        stringout = ' ' * key[0] + ' * %s [list]:\n' % key[1]
        keys = dset.keys()
        keys.sort()
        for k in keys:
            stringout += _format((key[0] + indent, ''), dset[k])
        return stringout

    def _format_tuple(key, dset):
        stringout = ' ' * key[0] + ' * %s [tuple]:\n' % key[1]
        keys = dset.keys()
        keys.sort()
        for k in keys:
            stringout += _format((key[0] + indent, ''), dset[k])
        return stringout

    def _format_arraytuple(key, dset):
        a = dset[...]
        if len(a) < 5:
            stringout = ' ' * key[0] + ' * ' + key[1] + ' [tuple = ' + str(tuple(a.ravel())) + ']\n'
        else:
            try:
                float(a.ravel()[0])
                stringout = ' ' * key[0] + ' * ' + key[1] + ' [tuple = (' + (
                            ('%f, ' * 4) % tuple(a.ravel()[:4])) + ' ...)]\n'
            except ValueError:
                stringout = ' ' * key[0] + ' * ' + key[1] + ' [tuple = (%d x %s objects)]\n' % (a.size, str(a.dtype))
        return stringout

    def _format_arraylist(key, dset):
        a = dset[...]
        if len(a) < 5:
            stringout = ' ' * key[0] + ' * ' + key[1] + ' [list = ' + str(a.tolist()) + ']\n'
        else:
            try:
                float(a.ravel()[0])
                stringout = ' ' * key[0] + ' * ' + key[1] + ' [list = [' + (
                            ('%f, ' * 4) % tuple(a.ravel()[:4])) + ' ...]]\n'
            except ValueError:
                stringout = ' ' * key[0] + ' * ' + key[1] + ' [list = [%d x %s objects]]\n' % (a.size, str(a.dtype))
        return stringout

    def _format_numpy(key, dset):
        a = dset[...]
        if len(a) < 5 and a.ndim == 1:
            stringout = ' ' * key[0] + ' * ' + key[1] + ' [array = ' + str(a.ravel()) + ']\n'
        else:
            stringout = ' ' * key[0] + ' * ' + key[1] + ' [' + (('%dx' * (a.ndim - 1) + '%d') % a.shape) + ' ' + str(
                a.dtype) + ' array]\n'
        return stringout

    def _format_scalar(key, dset):
        stringout = ' ' * key[0] + ' * ' + key[1] + ' [scalar = ' + str(dset[...]) + ']\n'
        return stringout

    def _format_str(key, dset):
        s = str(dset[...])
        if len(s) > 40:
            s = s[:40] + '...'
        stringout = ' ' * key[0] + ' * ' + key[1] + ' [string = "' + s + '"]\n'
        return stringout

    def _format_unicode(key, dset):
        s = str(dset[...]).decode('utf8')
        if len(s) > 40:
            s = s[:40] + '...'
        stringout = ' ' * key[0] + ' * ' + key[1] + ' [unicode = "' + s + '"]\n'
        return stringout

    def _format_pickle(key, dset):
        stringout = ' ' * key[0] + ' * ' + key[1] + ' [pickled object]\n'
        return stringout

    def _format_None(key, dset):
        stringout = ' ' * key[0] + ' * ' + key[1] + ' [None]\n'
        return stringout

    def _format_unknown(key, dset):
        stringout = ' ' * key[0] + ' * ' + key[1] + ' [unknown]\n'
        return stringout

    def _format(key, dset):
        dset_type = dset.attrs.get('type', None)

        # Treat groups as dicts
        if (dset_type is None) and (type(dset) is h5py.Group):
            dset_type = 'dict'

        if dset_type == 'dict':
            stringout = _format_dict(key, dset)
        elif dset_type == 'list':
            stringout = _format_list(key, dset)
        elif dset_type == 'array':
            stringout = _format_numpy(key, dset)
        elif dset_type == 'arraylist':
            stringout = _format_arraylist(key, dset)
        elif dset_type == 'tuple':
            stringout = _format_tuple(key, dset)
        elif dset_type == 'arraytuple':
            stringout = _format_arraytuple(key, dset)
        elif dset_type == 'string':
            stringout = _format_str(key, dset)
        elif dset_type == 'unicode':
            stringout = _format_unicode(key, dset)
        elif dset_type == 'scalar':
            stringout = _format_scalar(key, dset)
        elif dset_type == 'None':
            try:
                stringout = _format_numpy(key, dset)
            except:
                stringout = _format_None(key, dset)
        elif dset_type == 'pickle':
            stringout = _format_pickle(dset)
        elif dset_type is None:
            stringout = _format_numpy(key, dset)
        else:
            stringout = _format_unknown(key, dset)
        return stringout

    with h5py.File(filename, 'r') as f:
        h5rw_version = f.attrs.get('h5rw_version', None)
        #        if h5rw_version is None:
        #            print('Warning: this file does not seem to follow h5read format.')
        ctime = f.attrs.get('ctime', None)
        #        if ctime is not None:
        #            print('File created : ' + ctime)
        key_list = f.keys()
        outstring = ''
        for k in key_list:
            outstring += _format((0, k), f[k])

    print(outstring)

    # return string if output variable passed as option
    if output != None:
        return outstring




