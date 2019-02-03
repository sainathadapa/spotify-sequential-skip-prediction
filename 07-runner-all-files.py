import subprocess
import time
from glob import glob
import random


test_files = sorted(glob('./data/test_set/log_input_*.csv.gz'))
test_files = [x[-28:] for x in test_files]

commands_to_run = ['/home/sai/.conda/envs/myenv/bin/python 07-process-test-file.py {} {}'.format(x, y)
                   for x in test_files
                   for y in range(10, 21)]
random.shuffle(commands_to_run)

max_procs = 48
processes = []
while (len(processes) > 0) or (len(commands_to_run) > 0):

    if (len(processes) < max_procs) & (len(commands_to_run) > 0):
        this_cmd = commands_to_run.pop()
        print(this_cmd)
        this_proc = subprocess.Popen(this_cmd, shell=True)
        processes.append(this_proc)

    for i in range(len(processes)):
        this_proc = processes[i]
        if this_proc.poll() is not None:
            return_code = this_proc.poll()
            print('Process exited with code: ' + str(return_code))
            del processes[i]
            break

    time.sleep(1)
