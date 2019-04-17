import tvm
import numpy as np
from tvm.contrib import fbgemm
from tvm import autotvm
from collections import namedtuple
import sys
import logging

QuantParams = namedtuple("QuantParams", "scale zero_point") 

@autotvm.template
def test_fc_int8():
    n = 1024
    k = 1024
    m = 1024
    X = tvm.placeholder((m, k), name='X', dtype="int8")
    W = tvm.placeholder((n, k), name='W', dtype="int8")
    B = tvm.placeholder((n,), name='B', dtype="int")
    # quantization parameters will be got from Operator arguments
    X_qparams = QuantParams(scale=1.0, zero_point=0)
    W_qparams = QuantParams(scale=1.0, zero_point=0)
    Y_qparams = QuantParams(scale=1.0, zero_point=0)
    #MCB = 56
    #NCB = 32
    #KCB = 256
    #MR = 14
    #NR = 32
    #NR_MIN = 16 
    ROW_INTERLEAVE = 4
  
    MCBs = [48, 98, 144, 192, 240]
    NCBs = [16, 32, 64, 128, 48, 98, 192, 384]
    KCBs = [256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 960, 1024]
    MRs = [24, 12, 6, 3, 8, 4, 2, 1]
    NRs = [32]
    NR_MINs = [16]

    configs = autotvm.get_config()    
    configs.define_knob("MCBs", MCBs)
    configs.define_knob("NCBs", NCBs)
    configs.define_knob("KCBs", KCBs)
    configs.define_knob("MRs", MRs)
    configs.define_knob("NRs", NRs)
    configs.define_knob("NR_MINs", NR_MINs)
    configs.add_flop(2*m*n*k)

    C = fbgemm.fully_connected_int8(X, X_qparams, W, W_qparams, 
          B, Y_qparams, 1, True, 
          configs["MCBs"].val, 
          configs["NCBs"].val, 
          configs["KCBs"].val, 
          configs["MRs"].val, 
          configs["NRs"].val, 
          configs["NR_MINs"].val, 
          ROW_INTERLEAVE)

    Y = tvm.compute((m, n), lambda i, j: C[i][j], name="Y") 
    s = tvm.create_schedule(Y.op)
    return s, [X, W, B, Y]

if __name__ == "__main__":
    task = autotvm.task.create(test_fc_int8, args=(), target='llvm')
    print(task.config_space)
    # logging config (for printing tuning log to the screen)
    logging.getLogger('autotvm').setLevel(logging.DEBUG)
    logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

    # There are two steps for measuring a config: build and run.
    # By default, we use all cpu cores to compile program. Then measure them sequentially.
    # We measure 5 times and take average to reduce variance.
    measure_option = autotvm.measure_option(
        builder='local',
        runner=autotvm.LocalRunner(number=5))


    # begin tuning, log records to file `matmul.log`
    tuner = autotvm.tuner.RandomTuner(task)
    tuner.tune(n_trial=10,
           measure_option=measure_option,
           callbacks=[autotvm.callback.log_to_file('matmul.log')])

    #test_fc_int8()
