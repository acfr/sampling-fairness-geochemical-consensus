"""
MSF-GC. This module supports multiprocessing
Copyright (C) 2022  Raymond Leung

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Further information about this program can be obtained from:
- Raymond Leung (raymond.leung@sydney.edu.au)
"""

import numpy as np
from multiprocessing import Process, Queue
from algorithm_implementation import GradeBlockReliabilityAssessment


#Implement multiprocessing using a blocking, thread-safe queue
#that stores the returned values from child processes.
#Courtesy of sudo (https://stackoverflow.com/a/45829852)
#
class Multiprocessor():

    def __init__(self):
        self.processes = []
        self.queue = Queue()

    @staticmethod
    def _wrapper(func, queue, args, kwargs):
        ret = func(*args, **kwargs)
        queue.put(ret)

    def run(self, func, *args, **kwargs):
        args2 = [func, self.queue, args, kwargs]
        p = Process(target=self._wrapper, args=args2)
        self.processes.append(p)
        p.start()

    def wait(self):
        rets = []
        for p in self.processes:
            ret = self.queue.get()
            rets.append(ret)
        for p in self.processes:
            p.join()
        return rets


#Define worker method in this module for multiprocessing.Pool
#Rationale: https://stackoverflow.com/questions/41385708  To eliminate
#AttributeError: can't get attribute 'f' on <module '__main__' (built-in)>
#
def worker(procnum, num_processes, df, cfg):
    '''
    Evaluate geochemical consensus on a chunk of data.
    This method is used in validation.ipynb which executes the full procedure.
    '''
    num_rows = len(df)
    sample_size = cfg['sample_size']
    chunk_size = int(np.ceil((num_rows / num_processes) / sample_size) * sample_size)
    rows = np.arange(procnum * chunk_size, min((procnum+1) * chunk_size, num_rows))
    cols = ['component(c1)', 'component(c2)', 'component(c3)']
    print('process {} processing records {}-{}'.format(procnum, min(rows), max(rows)))

    eta_outliers = []
    rd_gmean = []
    rd_score = []
    consensus = []
    results = None
    sample_indices = np.unique(df.iloc[rows]['sample_id'].values)

    #'handle_breakdown' option is set upon instantiation
    gbr = GradeBlockReliabilityAssessment(cfg)
    for sample in sample_indices:
        chemistry = df[df['sample_id']==sample][cols].values
        out = gbr.compute_geochemical_consensus(chemistry, randseeds=[8365])[:-1]
        eta_outliers.append(out[0])
        rd_gmean.append(out[1])
        rd_score.append(out[2])
        consensus.append(out[3])

    return (procnum, np.c_[eta_outliers, rd_gmean, rd_score, consensus])
