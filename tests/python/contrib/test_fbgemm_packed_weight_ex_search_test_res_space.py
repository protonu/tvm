import tvm
import numpy as np
from tvm.contrib import fbgemm
from collections import namedtuple
from tvm import autotvm
import sys
import logging
import os

#raw_input("dummy breakpoint")
QuantParams = namedtuple("QuantParams", "scale zero_point")

def isValidConfig(mcb, ncb, kcb, mr, nr, nr_min, row_interleave):
    if (mcb % mr):
        return False
    if (ncb % nr):
        return False
    if (mr * (ncb/nr) > 24):
        return False
    return True


@autotvm.template
def tune_fbgemm_packed_weights(m, n, k):

   # The default values for FBGEMM without autotuning are:
   # MCBs = [56]
   # NCBs = [32]
   # KCBs = [256]
   # MRs = [14]
   # NRs = [32]
   # NR_MINs = [16]

    MCBs = [48, 98, 144, 192, 240]
    NCBs = [16, 32, 64, 128, 48, 98, 192, 384]
    KCBs = [256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 960, 1024]
    MRs = [24, 12, 6, 3, 8, 4, 2, 1]
    NRs = [16, 32]
    NR_MINs = [16]
    ROW_INTERLEAVE = 4

    valid_configs = []
    for mcb in MCBs:
        for ncb in NCBs:
            for kcb in KCBs:
                for mr in MRs:
                    for nr in NRs:
			# We don't need to check its validity using isValidConfig() because 
			# the pruning of search space is now within TVM itself.
		        valid_configs.append((mcb, ncb, kcb, mr, nr, 16, 4))


    #adding the default search point
    valid_configs.append((56,32,256,14,32,16,4))

    configs = autotvm.get_config()
    validate_func = lambda conf: isValidConfig(conf[0], conf[1], conf[2], conf[3], conf[4], conf[5], conf[6])
    configs.define_knob("VAL_CNFG", valid_configs, validate_func=validate_func)
    configs.add_flop(2 * m * n * k)

    ctx = tvm.cpu(0)
    W = tvm.placeholder((k, n), name='W', dtype="uint8")
    w = tvm.nd.array(np.random.uniform(1, 1, size=(k, n)).astype(W.dtype), ctx)

    my_packedw = tvm.get_global_func("tvm.contrib.fbgemm.pack_matrixB_int8")
    ww = my_packedw(w, 1,
                    configs["VAL_CNFG"].val[0],
                    configs["VAL_CNFG"].val[1],
                    configs["VAL_CNFG"].val[2],
                    configs["VAL_CNFG"].val[3],
                    configs["VAL_CNFG"].val[4],
                    configs["VAL_CNFG"].val[5],
		    ROW_INTERLEAVE)

    get_co_offsets = tvm.get_global_func(
        "tvm.contrib.fbgemm.compute_col_offsets_int8")
    co = get_co_offsets(w, 1, 1)

    X = tvm.placeholder((m, k), name='X', dtype="int8")
    B = tvm.placeholder((n,), name='B', dtype="int")

    # quantization parameters will be got from Operator arguments
    X_qparams = QuantParams(scale=1.0, zero_point=0)
    W_qparams = QuantParams(scale=1.0, zero_point=0)
    Y_qparams = QuantParams(scale=1.0, zero_point=0)

    C = fbgemm.gemm_int8acc32_prepacked_for_tuning(m, n, W,
                                        X, X_qparams,
                                        ww, W_qparams,
                                        B, Y_qparams, co, 1, True,
                                        configs["VAL_CNFG"].val[0],
                                        configs["VAL_CNFG"].val[1],
                                        configs["VAL_CNFG"].val[2],
                                        configs["VAL_CNFG"].val[3],
                                        configs["VAL_CNFG"].val[4],
                                        configs["VAL_CNFG"].val[5],
		                        ROW_INTERLEAVE)

    s = tvm.create_schedule(C.op)
    return s, [X, W, B, C]


if __name__ == "__main__":
    shapes = (
        [64, 800, 320],
        [64, 768, 512],
        [16, 256, 512],
        [128, 128, 128],
        [256, 512, 256],
        [1024, 1024, 1024])

    shapes_others = (
        [156800,    4,    36],
        [156800,    8,    36],
        [156800,    16,    36],
        [1,    128,    512],
        [1,    1024,    256],
        [1,    2048,   512],
        [1,    4096,    1024],
        [6,    256,    1024],
        [6,    256,    2048],
        [6,    512,    512],
        [6,    1024,    256],
        [6,    2048,    256],
        [6,    2048,    512],
        [6,    4096,    256],
        [6,    4096,    1024],
        [6,    4096,    2048],
        [10,    2048,    256],
        [10,    4096,    1024],
        [20,    2048,    256],
        [20,    4096,    1024],
        [102,    1024,    512],
        [102,    2323,    256],
        [102,    512,    256],
        [1,    800,    3200],
        [1,    800,    8000],)
        [16,    256,    1500],
        [16,    256,    1567],
        [1,    128,    2876]
        [16,    128,    1567]
        [1,    128,    2722]
	)
    
    # Running all shapes in shapes_others.
    # Warning: running all shapes in shapes_others can be very slow.
    if False:
         for shape in shapes_others:
              fbgemm_packed_weights(shape[0], shape[1], shape[2])

    else:
         for shape in shapes_others:
              task = autotvm.task.create(
                  tune_fbgemm_packed_weights, args=(
                      shape[0] , shape[1] , shape[2] ), target='llvm')
              # logging config (for printing tuning log to the screen)
              logging.getLogger('autotvm').setLevel(logging.DEBUG)
              logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

              measure_option = autotvm.measure_option(
                  builder='local',
                  runner=autotvm.LocalRunner(number=10, timeout=100000))

              tuner = autotvm.tuner.RandomTuner(task)
              log_file_name = "fbgemm_results_"+str(shape[0])+"_"+str(shape[1])+"_"+str(shape[2])+".log"
              tuner.tune(n_trial=400,
                         measure_option=measure_option,
                         callbacks=[autotvm.callback.log_to_file(log_file_name)])
