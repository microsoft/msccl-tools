import csv
import os
import re

home = os.getcwd()
SMS = 80
CHANNLES = 32

GPUS = 8
machine = 'a100'

def mpirun(collective, gpus, xml, txt, lower='384B', upper='3GB'):
    cmd = f'mpirun -np {gpus} -x NCCL_DEBUG=INFO -x NCCL_ALGO=RING,TREE,SCCL -x LD_LIBRARY_PATH={home}/msccl/build/lib/ ' \
        f'-x SCCL_XML_FILES={xml} -x NCCL_PROTO=SIMPLE,LL128,LL {home}/nccl-tests/build/{collective}_perf ' \
        f'-g 1 -n 100 -w 50 -f 2 -c 1 -z 0 -b {lower} -e {upper} > {txt}'
    print(f'Running {cmd}')
    os.system(cmd)

def mpirun_no_channel(collective, gpus, txt, lower='384B', upper='3GB', algo='RING,TREE'):
    cmd = f'mpirun -np {gpus} -x NCCL_DEBUG=INFO -x NCCL_ALGO={algo} -x LD_LIBRARY_PATH={home}/msccl/build/lib/ ' \
        f'-x NCCL_PROTO=SIMPLE,LL128,LL {home}/nccl-tests/build/{collective}_perf ' \
        f'-g 1 -n 100 -w 50 -f 2 -c 1 -z 0 -b {lower} -e {upper} > {txt}'
    print(f'Running {cmd}')
    os.system(cmd)

def allgather_ring(protocols, chans, insts):
    for protocol in ['LL', 'LL128', 'Simple']:
        for chan in [1]:
            for instances in [1, 12, 24]:
                if chan * instances < 32:
                    xml = f"{home}/xmls/allgather/ring_{chan}_{instances}_{protocol}.xml"
                    txt = f"{home}/{machine}/allgather/ring_{chan}_{instances}_{protocol}.txt"
                    print(f'Generating {xml} {txt}')
                    cmd = f'python3 sccl/examples/scclang/allgather_ring.py {GPUS} {chan} {instances} --protocol={protocol} > {xml}'
                    print(f'Running {cmd}')
                    os.system(cmd)
                    mpirun('all_gather', GPUS, xml, txt)

    for protocol in ['LL', 'LL128', 'Simple']:
        for chan in [8]:
            for instances in [4]:
                if chan * instances < 32:
                    xml = f"{home}/xmls/allgather/ring_{chan}_{instances}_{protocol}.xml"
                    txt = f"{home}/{machine}/allgather/ring_{chan}_{instances}_{protocol}.txt"
                    print(f'Generating {xml} {txt}')
                    cmd = f'python3 sccl/examples/scclang/allgather_ring.py {GPUS} {chan} {instances} --protocol={protocol} > {xml}'
                    print(f'Running {cmd}')
                    os.system(cmd)
                    mpirun('all_gather', GPUS, xml, txt, lower='256B', upper='4GB')
    
    for protocol in ['LL', 'LL128', 'Simple']:
        for chan in [8]:
            for instances in [2, 3]:
                if chan * instances < 32:
                    xml = f"{home}/xmls/allgather/ring_{chan}_{instances}_{protocol}.xml"
                    txt = f"{home}/{machine}/allgather/ring_{chan}_{instances}_{protocol}.txt"
                    print(f'Generating {xml} {txt}')
                    cmd = f'python3 sccl/examples/scclang/allgather_ring.py {GPUS} {chan} {instances} --protocol={protocol} > {xml}'
                    print(f'Running {cmd}')
                    os.system(cmd)
                    mpirun('all_gather', GPUS, xml, txt)

def allgather_recursive_doubling():
    protocol = 'LL'
    for instances in [1, 2, 4, 6]:
        xml = f"{home}/xmls/allgather/rd_{instances}.xml"
        txt = f"{home}/{machine}/allgather/rd_{instances}.txt"
        print(f'Generating {xml} {txt}')
        cmd = f'python3 sccl/examples/scclang/allgather_recursive_doubling.py {GPUS} {instances} --protocol {protocol} > {xml}'
        print(f'Running {cmd}')
        os.system(cmd)
        mpirun('all_gather', GPUS, xml, txt)


def allreduce_ring():
    for protocol in ['LL', 'LL128', 'Simple']:
        for chan in [1]:
            for instances in [1, 12, 24]:
                if chan * instances < 32:
                    xml = f"{home}/xmls/allreduce/ring_{chan}_{instances}_{protocol}.xml"
                    txt = f"{home}/{machine}/allreduce/ring_{chan}_{instances}_{protocol}.txt"
                    print(f'Generating {xml} {txt}')
                    cmd = f'python3 sccl/examples/scclang/allreduce_a100_ring.py {GPUS} {chan} {instances} --protocol={protocol} > {xml}'
                    print(f'Running {cmd}')
                    os.system(cmd)
                    mpirun('all_reduce', GPUS, xml, txt)

    for protocol in ['LL', 'LL128', 'Simple']:
        for chan in [8]:
            for instances in [4]:
                if chan * instances <= 32:
                    xml = f"{home}/xmls/allreduce/ring_{chan}_{instances}_{protocol}.xml"
                    txt = f"{home}/{machine}/allreduce/ring_{chan}_{instances}_{protocol}.txt"
                    print(f'Generating {xml} {txt}')
                    cmd = f'python3 sccl/examples/scclang/allreduce_a100_ring.py {GPUS} {chan} {instances} --protocol={protocol} > {xml}'
                    print(f'Running {cmd}')
                    os.system(cmd)
                    mpirun('all_reduce', GPUS, xml, txt, lower='256B', upper='4GB')

    for protocol in ['LL', 'LL128', 'Simple']:
        for chan in [8]:
            for instances in [2, 3]:
                if chan * instances < 32:
                    xml = f"{home}/xmls/allreduce/ring_{chan}_{instances}_{protocol}.xml"
                    txt = f"{home}/{machine}/allreduce/ring_{chan}_{instances}_{protocol}.txt"
                    print(f'Generating {xml} {txt}')
                    cmd = f'python3 sccl/examples/scclang/allreduce_a100_ring.py {GPUS} {chan} {instances} --protocol={protocol} > {xml}'
                    print(f'Running {cmd}')
                    os.system(cmd)
                    mpirun('all_reduce', GPUS, xml, txt)

def allreduce_recursive_doubling_halving():
    protocol='LL'
    for instances in [1, 2]:
        xml = f"{home}/xmls/allreduce/recursive_doubling_halving_{instances}_{protocol}.xml"
        txt = f"{home}/{machine}/allreduce/recursive_doubling_halving_{instances}_{protocol}.txt"
        print(f'Generating {xml} {txt}')
        cmd = f'python3 sccl/examples/scclang/allreduce_recursive_doubling_halving.py {GPUS} {instances} --protocol={protocol} > {xml}'
        print(f'Running {cmd}')
        os.system(cmd)
        mpirun('all_reduce', GPUS, xml, txt)

def allreduce_binomial_tree():
    protocol='LL'
    for trees in [1]:
        for instances in [1, 2]:
            xml = f"{home}/xmls/allreduce/binomial_tree_{trees}_{instances}_{protocol}.xml"
            txt = f"{home}/{machine}/allreduce/binomial_tree_{trees}_{instances}_{protocol}.txt"
            print(f'Generating {xml} {txt}')
            cmd = f'python3 sccl/examples/scclang/allreduce_binomial_tree.py {GPUS} {trees} {instances} --protocol={protocol} > {xml}'
            print(f'Running {cmd}')
            os.system(cmd)
            mpirun('all_reduce', GPUS, xml, txt)

def allgather_nccl():
    print("Run NCCL")
    mpirun_no_channel('all_gather', GPUS, f'{home}/{machine}/allgather/nccl.txt')

def allreduce_nccl():
    print("Run NCCL")
    mpirun_no_channel('all_reduce', GPUS, f'{home}/{machine}/allreduce/nccl.txt')

def parse(filename):
    parts = filename.split('.')
    # output = parts[0] + '.' + parts[1] + '.csv'
    output = parts[0] + '.csv'
    print("Reading log", filename)
    using_sccl = False
    labels = []
    results = []
    with open(filename, 'r') as f:
        
        line = f.readline()
        if "stdout" in line:
            stdout = True
            result_str = r"^\[1,0\]<stdout>:\s+[0-9]+\s+[0-9]+"
        else:
            stdout = False
            result_str = r"^\s+[0-9]+\s+[0-9]+"
        while line:
            if re.search(r"NCCL\sINFO\sConnected\s[0-9]+\sSCCL\salgorithms", line):
                using_sccl = True
            elif re.search(r"size\s+count", line):
                labels = line.split()[1:]
            elif re.search(result_str, line):
                nums = line.split()
                if stdout:
                    results.append(nums[1:])
                else:
                    results.append(nums)
            line = f.readline()

    if using_sccl:
        print("Using SCCL")
    else:
        print("Using NCCL")

    with open(output, 'w') as f:
        if len(results[0]) == 11:
            f.write("size,count,type,oop-time,oop-algbw,oop-busbw,oop-error,ip-time,ip-algbw,ip-busbw,ip-error\n")
        elif len(results[0]) == 12:
            f.write("size,count,type,op,oop-time,oop-algbw,oop-busbw,oop-error,ip-time,ip-algbw,ip-busbw,ip-error\n")


        writer = csv.writer(f)
        writer.writerows(results)

def allpairs():
    for ins in [1, 2]:
        xml = f"{home}/sccl/ap{ins}_nop.xml"
        txt = f"{home}/{machine}/allreduce/ap{ins}_nop.xml.txt"
        mpirun('all_reduce', GPUS, xml, txt, lower='384B', upper='32MB')


def check_create(dirname):
    if not os.path.exists(dirname):
        os.mkdir(dirname)

def extra_experiments():
    for protocol in ['LL128', 'Simple']:
        for chan in [8]:
            for instances in [4]:
                if chan * instances <= 32:
                    xml = f"{home}/xmls/allreduce/ring_{chan}_{instances}_{protocol}.xml"
                    txt = f"{home}/{machine}/allreduce/ring_{chan}_{instances}_{protocol}.txt"
                    print(f'Generating {xml} {txt}')
                    cmd = f'python3 sccl/examples/scclang/allreduce_a100_ring.py {GPUS} {chan} {instances} --protocol={protocol} > {xml}'
                    print(f'Running {cmd}')
                    os.system(cmd)
                    mpirun('all_reduce', GPUS, xml, txt, lower='256B', upper='4GB')

    for ins in [1, 2]:
        xml = f"{home}/sccl/ap{ins}_nop.xml"
        txt = f"{home}/{machine}/allreduce/ap{ins}_nop.xml.txt"
        mpirun('all_reduce', GPUS, xml, txt, lower='256B', upper='32MB')

    mpirun_no_channel('all_reduce', GPUS, f'{home}/{machine}/allreduce/nccl.txt', lower='256B', upper='32MB')
    

if __name__ == '__main__':
    check_create(f'{machine}')
    check_create(f'{machine}/allreduce')
    check_create(f'{machine}/allgather')
    check_create(f'xmls')
    check_create(f'xmls/allreduce')
    check_create(f'xmls/allgather')

    # allpairs()
    extra_experiments()

    # allgather_ring()
    # allgather_recursive_doubling()
    # allreduce_ring()
    # allreduce_recursive_doubling_halving()
    # allreduce_binomial_tree()

    # allgather_nccl()
    # allreduce_nccl()



    # for directory in [f'{home}/dgx2/allreduce', f'{home}/dgx2/allgather']:
    #     for filename in os.listdir(directory):
    #         f = os.path.join(directory, filename)
    #         # checking if it is a file
    #         if os.path.isfile(f) and f.endswith('.txt'):
    #             parse(f)
    

