# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections import defaultdict
import math
import humanfriendly

# The plans are keyed by (collective, machine_type) and each entry is a tuple
# (name, function, machines, size_ranges, priority).
synthesis_plans = defaultdict(list)


def register_synthesis_plan(collective, machine_type, machines=lambda x: True, sizes=None, priority=0):
    def decorator(fun):
        # Parse size_ranges
        size_ranges = []

        def parse_sizes(x):
            lower, upper = x
            if isinstance(lower, str):
                lower = humanfriendly.parse_size(lower)
            if isinstance(upper, str):
                upper = humanfriendly.parse_size(upper)
            if upper == None:
                upper = math.inf
            return (lower, upper)

        if sizes == None:
            size_ranges.append((0, math.inf))
        elif isinstance(sizes, list):
            for x in sizes:
                size_ranges.append(parse_sizes(x))
        else:
            size_ranges.append(parse_sizes(sizes))
        # Register entries under all keys that might trigger this plan
        entry = (fun.__name__, fun, machines, size_ranges, priority)
        if isinstance(machine_type, list):
            for mtype in machine_type:
                synthesis_plans[(collective, mtype)].append(entry)
        else:
            synthesis_plans[(collective, machine_type)].append(entry)
        # Return the original function to not break other use
        return fun
    return decorator
