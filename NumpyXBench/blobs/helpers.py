from ..utils.common import *

__all__ = ['get_dtypes']


# helper functions
def get_dtypes(dtypes):
    if not (isinstance(dtypes, str) or isinstance(dtypes, list)):
        raise AttributeError("Dtypes must be a string or list!")
    elif isinstance(dtypes, str):
        if dtypes in ['real_types', 'RealTypes']:
            return RealTypes
        elif dtypes in ['all_types', 'AllTypes']:
            return AllTypes
        else:
            raise AttributeError('Unknown dtypes name!')
    else:
        return dtypes
