# This utils is used to support Using pytorch's native DataParallel method,
# which create several backbone model inside DataParallel.
# DistributedDataParallel doesn't need this function.
from opendelta.utils.decorate import decorate
from collections import OrderedDict

def new_replicate_for_data_parallel(self):
    r""" self is the parent module. 
    """
    # rewrite the replicate in DataParallel.
    def _caller(_org_func, org_module, delta_name,  *args, **kwargs):
        args = args[1:] # the first argument here is ``self``
        delta_module = getattr(org_module, delta_name)
        if hasattr(delta_module, "pre_forward"):
            args, kwargs = delta_module.pre_forward(*args, **kwargs)
        ret = _org_func(*args, **kwargs)
        if hasattr(delta_module, "post_forward"):
            ret = delta_module.post_forward(ret)
        return ret
    replica = self.__new__(type(self))
    org_forward = replica.forward
    replica.__dict__ = self.__dict__.copy()
    assert replica.forward != org_forward
    replica.__dict__['forward'] = org_forward


    for _delta_info in self._delta_infos:
        if _delta_info['method'] == "insert_sequential" and _delta_info['state'] == "on":
            new_forward = decorate(replica.forward, _caller, extras=(replica, _delta_info['delta_name']), kwsyntax=True)
            replica.__dict__['forward'] = new_forward.__get__(replica, type(replica)) 
    
    # replicas do not have parameters themselves, the replicas reference the original
    # module.
    replica._parameters = OrderedDict()
    replica._buffers = replica._buffers.copy()
    replica._modules = replica._modules.copy()
    replica._is_replica = True

    return replica