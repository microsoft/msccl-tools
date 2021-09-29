# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections import defaultdict
import math
import tempfile
import os
import atexit
import humanfriendly

# The plans are keyed by (collective, machine_type) and each entry is a tuple
# (name, function, machines, size_range, protocol, priority).
synthesis_plans = defaultdict(list)


def _register_ef_provider(desc, fun, collective, machine_type, machines, sizes, protocol, priority):
    if sizes == None:
        sizes = (0, math.inf)
    else:
        lower, upper = sizes
        if isinstance(lower, str):
            lower = humanfriendly.parse_size(lower)
        if isinstance(upper, str):
            upper = humanfriendly.parse_size(upper)
        if upper == None:
            upper = math.inf
        sizes = (lower, upper)
    # Register entries under all keys that might trigger this plan
    entry = (desc, fun, machines, sizes, protocol, priority)
    if isinstance(machine_type, list):
        for mtype in machine_type:
            synthesis_plans[(collective, mtype)].append(entry)
    else:
        synthesis_plans[(collective, machine_type)].append(entry)


def register_ef_file(path, collective, machine_type, num_machines, sizes=None, protocol='Simple', priority=0):
    def provide_ef_path(machines):
        return path
    _register_ef_provider(f'load {path}', provide_ef_path, collective,
                         machine_type, lambda x: x == num_machines, sizes, protocol, priority)


def register_synthesis_plan(collective, machine_type, machines=lambda x: True, sizes=None, protocol='Simple', priority=0):
    def decorator(fun):
        def wrapped(machines):
            ef = fun(machines)
            fd, path = tempfile.mkstemp()
            with os.fdopen(fd, 'w') as f:
                f.write(ef)
            atexit.register(os.remove, path)
            return path
        _register_ef_provider(f'call {fun.__name__}', wrapped, collective,
                             machine_type, machines, sizes, protocol, priority)
        # Return the original function to not break other usage
        return fun
    return decorator
