# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from sccl.autosynth.registry import register_synthesis_plan

def register_a100_plans():
    @register_synthesis_plan('alltoall', 'a100', machines=lambda x: x == 9)
    def synthesize_a100_hierarchical_alltoall(machines, size):
        xml = ""
        nnodes = 9
        assert(machines == nnodes)
        ngpuspernode = 8
        instances = 2
        nchunksperloop = nnodes*ngpuspernode*instances
        xml += ('<algo name="test" nchunksperloop="{}" nchannels="{}" proto="Simple">'.format(nchunksperloop, 2*instances)) + '\n'

        def CrossNodeNghr(node, g):
            nghrNode = g if node > g else g+1
            nghrG = node if nghrNode > node else node-1
            return nghrNode, nghrG, nghrNode * ngpuspernode + nghrG
        for node in range(nnodes):
            for g in range(ngpuspernode):
                tbindex = 0
                nghrNode, nghrG, crossnodenghr = CrossNodeNghr(node,g)
                xml += ('  <gpu id="{}" i_chunks="{}" o_chunks="{}" s_chunks="{}">'.format(node*ngpuspernode+g, nchunksperloop, nchunksperloop, instances*2*ngpuspernode**2)) + '\n'
                for ch in range(instances):
                    xml += ('    <tb id="{}" send="{}" recv="-1" chan="{}">'.format(tbindex, crossnodenghr, ch)) + '\n'
                    xml += ('      <step s="0" type="s" srcbuf="s" srcoff="{}" dstbuf="s" dstoff="{}" cnt="{}" depid="{}" deps="{}" hasdep="0"/>'.format(ch*ngpuspernode**2, instances*ngpuspernode**2+ch*ngpuspernode**2, ngpuspernode**2, instances*(2+2*g)+ch, ngpuspernode)) + '\n'
                    xml += ('    </tb>') + '\n'
                    tbindex+=1
                for ch in range(instances):
                    xml += ('    <tb id="{}" send="-1" recv="{}" chan="{}">'.format(tbindex, crossnodenghr, ch)) + '\n'
                    xml += ('      <step s="0" type="r" srcbuf="s" srcoff="{}" dstbuf="s" dstoff="{}" cnt="{}" depid="-1" deps="-1" hasdep="1"/>'.format(ch*ngpuspernode**2, instances*ngpuspernode**2+ch*ngpuspernode**2, ngpuspernode**2)) + '\n'
                    xml += ('    </tb>') + '\n'
                    tbindex+=1
                for withinnodenghr  in range(ngpuspernode):
                    withinNghrNode, withinNghrG, withinCrossNodeNghr = CrossNodeNghr(node, withinnodenghr)
                    if withinnodenghr == g:
                        for ch in range(instances):
                            step = 0
                            xml += ('    <tb id="{}" send="-1" recv="-1" chan="0">'.format(tbindex)) + '\n'
                            xml += ('      <step s="{}" type="cpy" srcbuf="i" srcoff="{}" dstbuf="s" dstoff="{}" cnt="{}" depid="-1" deps="-1" hasdep="{}"/>'.format(step, instances*nghrNode*ngpuspernode+ch*ngpuspernode, instances*g*ngpuspernode+ch*ngpuspernode, ngpuspernode, 1)) + '\n'
                            step += 1
                            for j in range(ch*(ngpuspernode//instances), (ch+1)*(ngpuspernode//instances)):
                                for k in range(instances):
                                    xml += ('      <step s="{}" type="nop" srcbuf="i" srcoff="0" dstbuf="o" dstoff="0" cnt="0" depid="{}" deps="{}" hasdep="{}"/>'.format(step, (instances*(2*j+2+1)+k) if j < g else (instances*(2*j+2)+k), 0, 1 if step == 1+ngpuspernode-1 else 0)) + '\n'
                                    step += 1
                            xml += ('    </tb>') + '\n'
                            tbindex+=1
                    else:
                        for ch in range(instances):
                            xml += ('    <tb id="{}" send="{}" recv="-1" chan="{}">'.format(tbindex, node*ngpuspernode+withinnodenghr, ch)) + '\n'
                            xml += ('      <step s="0" type="s" srcbuf="i" srcoff="{}" dstbuf="s" dstoff="{}" cnt="{}" depid="-1" deps="-1" hasdep="0"/>'.format(instances*withinNghrNode*ngpuspernode+ch*ngpuspernode, instances*g*ngpuspernode+ch*ngpuspernode, ngpuspernode)) + '\n'
                            xml += ('    </tb>') + '\n'
                            tbindex+=1
                        for ch in range(instances):
                            xml += ('    <tb id="{}" send="-1" recv="{}" chan="{}">'.format(tbindex, node*ngpuspernode+withinnodenghr, ch)) + '\n'
                            xml += ('      <step s="0" type="r" srcbuf="i" srcoff="{}" dstbuf="s" dstoff="{}" cnt="{}" depid="-1" deps="-1" hasdep="1"/>'.format(instances*nghrNode*ngpuspernode+ch*ngpuspernode, instances*withinnodenghr*ngpuspernode+ch*ngpuspernode, ngpuspernode)) + '\n'
                            xml += ('    </tb>') + '\n'
                            tbindex+=1

        # --------------------------------
                for withinnodenghr  in range(ngpuspernode):
                    withinNghrNode, withinNghrG, withinCrossNodeNghr = CrossNodeNghr(node, withinnodenghr)
                    if withinnodenghr == g:
                        for ch in range(instances):
                            xml += ('    <tb id="{}" send="-1" recv="-1" chan="0">'.format(tbindex)) + '\n'
                            step = 0
                            xml += ('      <step s="{}" type="cpy" srcbuf="i" srcoff="{}" dstbuf="o" dstoff="{}" cnt="{}" depid="-1" deps="-1" hasdep="0"/>'.format(step, instances*(node*ngpuspernode+g)+ch, instances*(node*ngpuspernode+g)+ch, 1)) + '\n'
                            step += 1
                            for j in range(ngpuspernode):
                                xml += ('      <step s="{}" type="cpy" srcbuf="s" srcoff="{}" dstbuf="o" dstoff="{}" cnt="{}" depid="{}" deps="{}" hasdep="0"/>'.format(step, instances*(ngpuspernode**2+j*ngpuspernode+g)+ch, instances*(nghrNode*ngpuspernode+j)+ch, 1, instances+(instances*(j*ngpuspernode+g)+ch)//((instances*ngpuspernode**2)//instances), 0)) + '\n'
                                step += 1
                            xml += ('    </tb>') + '\n'
                            tbindex+=1
                    else:
                        for ch in range(instances):
                            xml += ('    <tb id="{}" send="{}" recv="-1" chan="{}">'.format(tbindex, node*ngpuspernode+withinnodenghr, instances+ch)) + '\n'
                            step = 0
                            xml += ('      <step s="{}" type="s" srcbuf="i" srcoff="{}" dstbuf="o" dstoff="{}" cnt="{}" depid="-1" deps="-1" hasdep="0"/>'.format(step, instances*(node*ngpuspernode+withinnodenghr)+ch, instances*(node*ngpuspernode+g)+ch, 1)) + '\n'
                            step += 1
                            for j in range(ngpuspernode):
                                xml += ('      <step s="{}" type="s" srcbuf="s" srcoff="{}" dstbuf="o" dstoff="{}" cnt="{}" depid="{}" deps="{}" hasdep="0"/>'.format(step, instances*(ngpuspernode**2+j*ngpuspernode+withinnodenghr)+ch, instances*(nghrNode*ngpuspernode+j)+ch, 1, instances+(instances*(j*ngpuspernode+withinnodenghr)+ch)//((instances*ngpuspernode**2)//instances), 0)) + '\n'
                                step += 1
                            xml += ('    </tb>') + '\n'
                            tbindex+=1
                        for ch in range(instances):
                            xml += ('    <tb id="{}" send="-1" recv="{}" chan="{}">'.format(tbindex, node*ngpuspernode+withinnodenghr, instances+ch)) + '\n'
                            step = 0
                            xml += ('      <step s="{}" type="r" srcbuf="i" srcoff="{}" dstbuf="o" dstoff="{}" cnt="{}" depid="-1" deps="-1" hasdep="0"/>'.format(step, instances*(node*ngpuspernode+g)+ch, instances*(node*ngpuspernode+withinnodenghr)+ch, 1)) + '\n'
                            step += 1
                            for j in range(ngpuspernode):
                                xml += ('      <step s="{}" type="r" srcbuf="s" srcoff="{}" dstbuf="o" dstoff="{}" cnt="{}" depid="-1" deps="-1" hasdep="0"/>'.format(step, instances*(ngpuspernode**2+j*ngpuspernode+g)+ch, instances*(withinNghrNode*ngpuspernode+j)+ch, 1)) + '\n'
                                step += 1
                            xml += ('    </tb>') + '\n'
                            tbindex+=1
                xml += ('  </gpu>') + '\n'
        xml += ('</algo>') + '\n'
        return xml