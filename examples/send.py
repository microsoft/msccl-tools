# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# This script defines and saves a custom collective to send from rank 2 to rank 7

from msccl.collectives import build_collective
from msccl.serialization import save_msccl_object

precondition = lambda r, c: r == 2
postcondition = lambda r, c: r == 7
coll = build_collective('Send', 8, 1, precondition, postcondition)
save_msccl_object(coll, 'send.json')
