"""Exact-key, proof-local geometry caching; never reuses a safety verdict."""
from collections import OrderedDict
from contextlib import contextmanager, ExitStack
from contextvars import ContextVar
from functools import wraps
import numpy as np
from .geometry import Capsule

_current = ContextVar('validation_geometry_cache', default=None)


def array_key(value):
    a=np.asarray(value,dtype=float)
    return a.shape,a.tobytes()


def copy_geometry(value):
    # Geometry has a small, known value vocabulary. Avoid generic deepcopy's
    # per-dataclass reflection while keeping cached arrays private.
    if isinstance(value,np.ndarray):return value.copy()
    if isinstance(value,Capsule):return Capsule(value.name,value.start.copy(),value.end.copy(),value.radius)
    if isinstance(value,dict):return {k:copy_geometry(v) for k,v in value.items()}
    if isinstance(value,list):return [copy_geometry(v) for v in value]
    if isinstance(value,tuple):return tuple(copy_geometry(v) for v in value)
    return value


@contextmanager
def geometry_cache(models):
    # A proof owns model state while caching; cache dies before releasing locks.
    if _current.get() is not None:
        yield
        return
    with ExitStack() as stack:
        for _,model in sorted(models.items()):
            if hasattr(model,'_lock'):stack.enter_context(model._lock)
        token=_current.set({'pose':{},'values':OrderedDict()})
        try:yield
        finally:_current.reset(token)


def cached_geometry(method):
    @wraps(method)
    def call(self,*args,**kwargs):
        cache=_current.get()
        if cache is None:return method(self,*args,**kwargs)
        if method.__name__!='frame':
            self.update(args[0])
        def key(x):return x if isinstance(x,(str,int,float,bool,type(None))) else array_key(x)
        identity=(id(self),method.__name__,cache['pose'].get(id(self)),tuple(key(x) for x in args),tuple((k,key(v)) for k,v in sorted(kwargs.items())))
        values=cache['values']
        if identity in values:
            values.move_to_end(identity)
            return copy_geometry(values[identity])
        value=method(self,*args,**kwargs)
        values[identity]=copy_geometry(value)
        if len(values)>512:values.popitem(last=False)
        return value
    return call
