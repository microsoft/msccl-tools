import csv
import os
import re

home = os.getcwd()
SMS = 80
CHANNLES = 32

def mpirun(collective, gpus, xml, txt, lower='128B', upper='128MB'):
    cmd = f'mpirun -np {gpus} -x NCCL_DEBUG=INFO -x NCCL_ALGO=RING,TREE,SCCL -x LD_LIBRARY_PATH={home}/msccl/build/lib/ ' \
        f'-x NCCL_MIN_CHANNELS=32 -x SCCL_XML_FILES={xml} -x NCCL_PROTO=SIMPLE,LL128,LL {home}/nccl-tests/build/{collective}_perf ' \
        f'-g 1 -n 100 -w 50 -f 2 -c 1 -z 0 -b {lower} -e {upper} > {txt}'
    print(f'Running {cmd}')
    os.system(cmd)

def mpirun_no_channel(collective, gpus, txt, lower='128B', upper='128MB'):
    cmd = f'mpirun -np {gpus} -x NCCL_DEBUG=INFO -x NCCL_ALGO=RING,TREE -x LD_LIBRARY_PATH={home}/msccl/build/lib/ ' \
        f'-x NCCL_PROTO=SIMPLE,LL128,LL {home}/nccl-tests/build/{collective}_perf ' \
        f'-g 1 -n 100 -w 50 -f 2 -c 1 -z 0 -b {lower} -e {upper} > {txt}'
    print(f'Running {cmd}')
    os.system(cmd)

def allgather_ring():
    protocol='LL128'
    for chan in [1, 8, 16]:
        for instances in [1, 2, 4, 8, 16, 32]:
            if chan * instances < 32:
                xml = f"{home}/xmls/allgather/ring_{chan}_{instances}_{protocol}.xml"
                txt = f"{home}/dgx2/allgather/ring_{chan}_{instances}_{protocol}.txt"
                print(f'Generating {xml} {txt}')
                cmd = f'python3 sccl/examples/scclang/allgather_ring.py 16 {chan} {instances} --protocol={protocol} > {xml}'
                print(f'Running {cmd}')
                os.system(cmd)
                mpirun('all_gather', 16, xml, txt)

def allgather_recursive_doubling():
    protocol = 'LL'
    for instances in [1, 2, 4, 8]:
        xml = f"{home}/xmls/allgather/rd_{instances}.xml"
        txt = f"{home}/dgx2/allgather/rd_{instances}.txt"
        print(f'Generating {xml} {txt}')
        cmd = f'python3 sccl/examples/scclang/allgather_recursive_doubling.py 16 {instances} --protocol {protocol} > {xml}'
        print(f'Running {cmd}')
        os.system(cmd)
        mpirun('all_gather', 16, xml, txt)


def allreduce_ring():
    protocol='LL128'
    for chan in [1, 16]:
        for instances in [1, 2, 16, 32]:
            if chan * instances <= 32:
                xml = f"{home}/xmls/allreduce/ring_{chan}_{instances}_{protocol}.xml"
                txt = f"{home}/dgx2/allreduce/ring_{chan}_{instances}_{protocol}.txt"
                print(f'Generating {xml} {txt}')
                cmd = f'python3 sccl/examples/scclang/allreduce_a100_ring.py 16 {chan} {instances} --protocol={protocol} > {xml}'
                print(f'Running {cmd}')
                os.system(cmd)
                mpirun('all_reduce', 16, xml, txt)

def allreduce_recursive_doubling_halving():
    protocol='LL'
    for instances in [1]:
        xml = f"{home}/xmls/allreduce/recursive_doubling_halving_{instances}_{protocol}.xml"
        txt = f"{home}/dgx2/allreduce/recursive_doubling_halving_{instances}_{protocol}.txt"
        print(f'Generating {xml} {txt}')
        cmd = f'python3 sccl/examples/scclang/allreduce_recursive_doubling_halving.py 16 {instances} --protocol={protocol} > {xml}'
        print(f'Running {cmd}')
        os.system(cmd)
        mpirun('all_reduce', 16, xml, txt)

def allreduce_binomial_tree():
    protocol='LL'
    for trees in [1]:
        for instances in [1]:
            xml = f"{home}/xmls/allreduce/binomial_tree_{trees}_{instances}_{protocol}.xml"
            txt = f"{home}/dgx2/allreduce/binomial_tree_{trees}_{instances}_{protocol}.txt"
            print(f'Generating {xml} {txt}')
            cmd = f'python3 sccl/examples/scclang/allreduce_binomial_tree.py 16 {trees} {instances} --protocol={protocol} > {xml}'
            print(f'Running {cmd}')
            os.system(cmd)
            mpirun('all_reduce', 16, xml, txt)

def allgather_nccl():
    print("Run NCCL")
    mpirun_no_channel('all_reduce', 16, '{home}/dgx2/allgather/nccl.txt')

def allreduce_nccl():
    print("Run NCCL")
    mpirun_no_channel('all_reduce', 16, '{home}/dgx2/allreduce/nccl.txt')

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
        f.write("size,count,type,oop-time,oop-algbw,oop-busbw,oop-error,ip-time,ip-algbw,ip-busbw,ip-error\n")

        writer = csv.writer(f)
        writer.writerows(results)

if __name__ == '__main__':
    # # allgather_ring()
    # # allgather_recursive_doubling()
    # allreduce_ring()
    allreduce_recursive_doubling_halving()
    allreduce_binomial_tree()

    # # allgather_nccl()
    # # allreduce_nccl()

    for directory in [f'{home}/dgx2/allreduce', f'{home}/dgx2/allgather']:
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            # checking if it is a file
            if os.path.isfile(f) and f.endswith('.txt'):
                parse(f)
    

