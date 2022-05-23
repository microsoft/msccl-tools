# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Example of a simple custom collective where Rank 0 sends a chunk to Ranks 1 and 2

from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import Collective

# For custom collectives you need to define a new collective class
# this is used by mscclang to initialize buffers with chunks (pre-condition)
# and provide a checker to check that chunks satisfy the post-condition of the collective.
class CollEx(Collective):
    # Initial state is chunk0 is on rank0 in the input buffer
    def init_buffers(self):
        chunks_per_node = self.chunk_factor
        rank_buffers = []
        for r in range(self.num_ranks):
            input_buffer = [None] * chunks_per_node
            output_buffer = [None] * chunks_per_node 
            if r == 0:
                for c in range(chunks_per_node):
                    # Format for specifying a chunk
                    # Chunk(starting rank, starting index, ending rank, ending index)
                    # Because this chunk ends up on multiple ranks ending rank is set to -1
                    input_buffer[c] = Chunk(r, c, -1, c)
            buffers = {Buffer.input : input_buffer, 
                       Buffer.output : output_buffer}
            rank_buffers.append(buffers)
        return rank_buffers
            

    # Final state chunk0 from rank0 is in the output buffer of rank1 and rank2
    def check(self, prog):
        correct = True
        for r in range(1, self.num_ranks):
            output = prog.buffers[r][Buffer.output]
            for c in range(self.chunk_factor):
                chunk = output[c]
                # Check that we got chunk 0 from rank 0
                if chunk is None or chunk.origin_rank != 0 or chunk.origin_index != 0:
                    print(f'Rank {r} chunk {c} is incorrect should be ({0}, {0}) given {chunk}')
                    correct = False
        return correct


def custom_example1():
    # MSCCLang programs take in a name for hte program, the topology of the network, 
    # collective being implemented, chunksperloop of the collective, and optionally the NCCL protocol to be used
    size = 3
    topology = fully_connected(size) 
    # Collectives take in number of ranks in the network, chunksperloop of the collective, whether it is inplace, 
    collective = CollEx(size, 1, inplace=False)
    with MSCCLProgram("allgather_ring", topology, collective, instances=1, protocol="Simple"):
        # Get the chunk at rank 0 index 0 of the input buffer
        c = chunk(0, Buffer.input, 0)
        # Send chunks to 1 and 2
        # Can specify the sender's tb, receiver's tb, and channel for the send operation
        # MSCCLang provides a default threadblock assignment if they aren't specified
        # MSCCLang will also check the tb/channel combos are valid
        c.copy(1, buffer=Buffer.output, index=0, sendtb=1, recvtb=1, ch=0)
        c.copy(2, buffer=Buffer.output, index=0, sendtb=2, recvtb=1, ch=1)

        XML() # Generates the XML for this collective
        Check() # Checks the routes defined for each chunk are correct. Currently doesn't check XML correct

def custom_example2():

    size = 3
    topology = fully_connected(size) 

    collective = CollEx(size, 1, inplace=False)
    with MSCCLProgram("allgather_ring", topology, collective, instances=1, protocol="Simple"):
        c = chunk(0, Buffer.input, 0)
        # This is the same program as above but instead of rank 0 sending to 1 and 2
        # 0 sends to 1 which sends to 2
        # send returns the chunk on the receiver's side
        c = c.copy(1, buffer=Buffer.output, index=0, sendtb=1, recvtb=1, ch=0)
        c.copy(2, buffer=Buffer.output, index=0, sendtb=2, recvtb=1, ch=1)

        XML()
        Check() 

custom_example1()
custom_example2()
