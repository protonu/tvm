import tvm
import numpy as np
from tvm.contrib import fbgemm
from collections import namedtuple
from tvm import autotvm
import sys
import logging
import os
import random
from conv_with_requant_ref import *

#raw_input("dummy breakpoint")
QuantParams = namedtuple("QuantParams", "scale zero_point")


@autotvm.template
def tune_fbgemm_packed_weights(m, n, k):

    MCBs = [56]
    NCBs = [32]
    KCBs = [256]
    MRs = [14]
    NRs = [32]
    NR_MINs = [16]

    ROW_INTERLEAVE = 4

    MCBs = [48, 98, 144, 192, 240]
    NCBs = [16, 32, 64, 128, 48, 98, 192, 384]
    KCBs = [256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 960, 1024]
    MRs = [24, 12, 6, 3, 8, 4, 2, 1]
    NRs = [16, 32]
    NR_MINs = [16]

    configs = autotvm.get_config()
    configs.define_knob("MCBs", MCBs)
    configs.define_knob("NCBs", NCBs)
    configs.define_knob("KCBs", KCBs)
    configs.define_knob("MRs", MRs)
    configs.define_knob("NRs", NRs)
    configs.define_knob("NR_MINs", NR_MINs)
    configs.add_flop(2 * m * n * k)

    ctx = tvm.cpu(0)
    W = tvm.placeholder((k, n), name='W', dtype="uint8")
    w = tvm.nd.array(np.random.uniform(1, 1, size=(k, n)).astype(W.dtype), ctx)

    my_packedw = tvm.get_global_func("tvm.contrib.fbgemm.pack_matrixB_int8")
    ww = my_packedw(w, 1,
                    configs["MCBs"].val,
                    configs["NCBs"].val,
                    configs["KCBs"].val,
                    configs["MRs"].val,
                    configs["NRs"].val,
                    configs["NR_MINs"].val,
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
                                        configs["MCBs"].val,
                                        configs["NCBs"].val,
                                        configs["KCBs"].val,
                                        configs["MRs"].val,
                                        configs["NRs"].val,
                                        configs["NR_MINs"].val,
                                        ROW_INTERLEAVE)

    s = tvm.create_schedule(C.op)
    #f = tvm.build(s, [X,W, B, C], target="llvm", name="packedmatmul")
    return s, [X, W, B, C]

def fbgemm_packed_weights(m, n, k):

    MCB = 56
    NCB = 32
    KCB = 256
    MR = 14
    NR = 32
    NR_MIN = 16
    ROW_INTERLEAVE = 4

    MCB = 48
    NCB = 16
    KCB = 640
    MR = 24
    NR = 16
    NR_MIN = 16
    ROW_INTERLEAVE = 4


    ctx = tvm.cpu(0)
    W = tvm.placeholder((k, n), name='W', dtype="uint8")
    w = tvm.nd.array(np.random.uniform(1, 1, size=(k, n)).astype(W.dtype), ctx)

    my_packedw = tvm.get_global_func("tvm.contrib.fbgemm.pack_matrixB_int8")
    ww = my_packedw(w, 1,
                    MCB,
                    NCB,
                    KCB,
                    MR,
                    NR,
                    NR_MIN,
		    ROW_INTERLEAVE)

    print_packed_b =tvm.get_global_func("tvm.contrib.fbgemm.print_packb") 
    #print_packed_b(ww)
 
    get_co_offsets = tvm.get_global_func(
        "tvm.contrib.fbgemm.compute_col_offsets_int8")
    co = get_co_offsets(w, 1, 1)

    X = tvm.placeholder((m, k), name='X', dtype="int8")
    B = tvm.placeholder((n,), name='B', dtype="int")

    # quantization parameters will be got from Operator arguments
    X_qparams = QuantParams(scale=1.0, zero_point=0)
    W_qparams = QuantParams(scale=1.0, zero_point=0)
    Y_qparams = QuantParams(scale=1.0, zero_point=0)

    C = fbgemm.gemm_int8acc32_prepacked(m, n,
                                        X, X_qparams,
                                        ww, W_qparams,
                                        B, Y_qparams, co, 1, True,
                                        MCB,
                                        NCB,
                                        KCB,
                                        MR,
                                        NR,
                                        NR_MIN,
		                        ROW_INTERLEAVE)

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [X, B, C], target="llvm", name="packedmatmul")

    x = tvm.nd.array(np.random.uniform(2, 2, size=(m, k)).astype(X.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(0, 0, size=(n,)).astype(B.dtype), ctx)
    y = tvm.nd.array(np.zeros((m, n), dtype=C.dtype), ctx)
    #f(x,b,y)

    f_evaluator = f.time_evaluator(f.entry_name, ctx, 10)
    result = f_evaluator(x,b,y)
    print(result)
    gops_per_mm = 2*m*n*k
    gops_per_sec = gops_per_mm/result.mean/1e9
    print("M:{}, N:{}, K:{}".format(m,n,k))
    print(gops_per_sec)
    tvm.testing.assert_allclose(
           y.asnumpy(), np.matmul(x.asnumpy(), w.asnumpy()) + b.asnumpy(), rtol=1e-5)

def test_fbgemm_packed_weights_with_requant(m, n, k, w_val, x_val, b_val):
    ctx = tvm.cpu(0)
    W = tvm.placeholder((k, n), name='W', dtype="uint8")
    w = tvm.nd.array(np.random.uniform(w_val - 1, w_val + 2, size=(k, n)).astype(W.dtype), ctx)
    my_packedw = tvm.get_global_func("tvm.contrib.fbgemm.pack_matrixB_int8")
    ww = my_packedw(w, 1)

    get_co_offsets = tvm.get_global_func("tvm.contrib.fbgemm.compute_col_offsets_int8")
    co = get_co_offsets(w, 1, 1)

    X = tvm.placeholder((m, k), name='X', dtype="int8")
    B = tvm.placeholder((n,), name='B', dtype="int")

    # quantization parameters will be got from Operator arguments
    X_qparams = QuantParams(scale=1.0, zero_point=0)
    W_qparams = QuantParams(scale=1.0, zero_point=0)
    Y_qparams = QuantParams(scale=1.0, zero_point=0)

    C = fbgemm.gemm_int8acc32_prepacked_with_requant(m, n, X, X_qparams, ww, W_qparams, B, Y_qparams, co)
    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [X, B, C], target="llvm", name="packedmatmul_with_requant")
    f_evaluator = f.time_evaluator(f.entry_name, ctx, 10)

    x = tvm.nd.array(np.random.uniform(x_val - 1, x_val + 2, size=(m, k)).astype(X.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(b_val - 1, b_val + 2, size=(n,)).astype(B.dtype), ctx)
    y = tvm.nd.array(np.zeros((m, n), dtype=C.dtype), ctx)
    f(x,b,y)
    tvm.testing.assert_allclose(
           y.asnumpy(), np.matmul(x.asnumpy(), w.asnumpy()) + b.asnumpy(), rtol=1e-5)

def test_fbgemm_packed_weights_with_requant_and_trans(m, n, k, w_val, x_val, b_val, A_trans, W_trans):
    ctx = tvm.cpu(0)

    # transposing W if it's transposed
    W = tvm.placeholder((k, n), name='W', dtype="uint8")
    w_un = np.random.uniform(w_val - 1, w_val + 2, size=(k, n)).astype(W.dtype)
    if W_trans:
        w = tvm.nd.array(w_un.transpose(), ctx)
    else:
        w = tvm.nd.array(w_un, ctx)

    my_packedw = tvm.get_global_func("tvm.contrib.fbgemm.pack_matrixB_int8")
    ww = my_packedw(w, 1, W_trans)

    get_co_offsets = tvm.get_global_func("tvm.contrib.fbgemm.compute_col_offsets_int8")
    co = get_co_offsets(w, 1, 1, True)

    B = tvm.placeholder((n,), name='B', dtype="int")

    # quantization parameters will be got from Operator arguments
    X_qparams = QuantParams(scale=1.0, zero_point=0)
    W_qparams = QuantParams(scale=1.0, zero_point=0)
    Y_qparams = QuantParams(scale=1.0, zero_point=0)

    C = fbgemm.gemm_int8acc32_prepacked_with_requant(m, n, X, X_qparams, ww, W_qparams,
						     B, Y_qparams, co, A_trans)

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [X, B, C], target="llvm", name="packedmatmul_with_requant")
    f_evaluator = f.time_evaluator(f.entry_name, ctx, 10)

    # transposing X if it's transposed
    if A_trans:
        X = tvm.placeholder((k, m), name='X', dtype="int8")
    else:
        X = tvm.placeholder((m, k), name='X', dtype="int8")
    x_un = np.random.uniform(x_val - 1, x_val + 2, size=(m, k)).astype(X.dtype)
    if A_trans:
        x = tvm.nd.array(x_un.transpose(), ctx)
    else:
        x = tvm.nd.array(x_un, ctx)

    b = tvm.nd.array(np.random.uniform(b_val, b_val + 2, size=(n,)).astype(B.dtype), ctx)
    y = tvm.nd.array(np.zeros((m, n), dtype=C.dtype), ctx)
    f(x,b,y)

    print("M:{}, N:{}, K:{}".format(m,n,k))
    tvm.testing.assert_allclose(
           y.asnumpy(), np.matmul(x_un, w_un) + b.asnumpy(), rtol=1e-5)


def test_fbgemm_conv_int8(spatial_dim, MB, IC, OC, IN_DIM_lst, G, K_lst, stride_lst, pad_lst):
    ctx = tvm.cpu(0)
    IN_DIM = tvm.nd.array(np.array(IN_DIM_lst).astype("int32"), ctx)
    K = tvm.nd.array(np.array(K_lst).astype("int32"), ctx)
    stride = tvm.nd.array(np.array(stride_lst).astype("int32"), ctx)
    pad = tvm.nd.array(np.array(pad_lst).astype("int32"), ctx)
    
    IN_DIMP = [0 for i in range(spatial_dim)]
    OUT_DIM = [0 for i in range(spatial_dim)]

    for d in range(spatial_dim):
      IN_DIMP[d] = IN_DIM_lst[d] + pad_lst[d] + pad_lst[spatial_dim + d];
      OUT_DIM[d] = (IN_DIMP[d] - K_lst[d]) / stride_lst[d] + 1;
    }

    # shapes
    input_shape = [MB] + IN_DIM_lst + [IC] #NHWC
    W_shape = K_lst + [IC, OC / G] #RSCK
    Y_shape = [MB] + OUT_DIM + [OC] #NHWK
    # weight
    W = tvm.placeholder(W_shape, name='W', dtype="int8")
    wa_length = np.prod(K_lst) * IC * OC / G
    wa = [random.randint(-4, 4) for i in range(wa_length)]
    w = tvm.nd.array(np.reshape(np.array(wa), W_shape).astype(W.dtype), ctx)
    
    # packing of weight
    my_packedw = tvm.get_global_func("tvm.contrib.fbgemm.pack_matrixB_int8_conv")
    
    ww = my_packedw(w, spatial_dim, MB, IC, OC, IN_DIM, G, K, stride, pad)
    
    # input (X)
    X = tvm.placeholder(input_shape, name='X', dtype="uint8")
    
    # quantization parameters will be got from Operator arguments
    X_zero_point = 4
    W_zero_point = -2
    create_pointer_vector_int = \
    tvm.get_global_func("tvm.contrib.fbgemm.create_pointer_vector_int")
    Y_zero_point = 5
    
    # column offset
    get_co_offsets = \
    tvm.get_global_func("tvm.contrib.fbgemm.compute_col_offsets_int8_conv")
    co = get_co_offsets(w, W_zero_point, spatial_dim,
                        MB, IC, OC, IN_DIM, G, K, stride, pad)

    C_multiplier = 0.0878014
    
    in_dim_v = create_pointer_vector_int(IN_DIM, spatial_dim)
    k_v = create_pointer_vector_int(K, spatial_dim)
    stride_v = create_pointer_vector_int(stride, spatial_dim)
    pad_v = create_pointer_vector_int(pad, spatial_dim * 2)

    C = fbgemm.conv_int8(Y_shape, X, X_zero_point, ww,
                         W_zero_point, Y_zero_point, C_multiplier, co,
                         MB, IC, OC, in_dim_v, G, k_v, stride_v, pad_v, spatial_dim)
    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [X, C], target="llvm", name="conv_int8")

    x_length = MB * np.prod(IN_DIM_lst) * IC
    xa = [random.randint(0, 5) for i in range(x_length)]
    x = tvm.nd.array(np.reshape(np.array(xa), input_shape).astype(X.dtype), ctx)
    y = tvm.nd.array(np.zeros(Y_shape, dtype=C.dtype), ctx)
    f(x,y)

    y_ref = reference_solution(xa, X_zero_point, wa, MB, IC, OC, IN_DIM_lst,
                               OUT_DIM, G, K_lst, stride_lst, pad_lst, [C_multiplier],
                               [W_zero_point], Y_zero_point)
    y_ref = np.reshape(np.array(y_ref), Y_shape)

    tvm.testing.assert_allclose(y.asnumpy(), y_ref, rtol=1e-5)



if __name__ == "__main__":
    shapes = (
        [64, 800, 320],
        [64, 768, 512],
        #[16, 256, 512],
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
        [1,    800,    8000],
        [16,    256,    1500],
        [16,    256,    1567],
        [1,    128,    2876],
        [16,    128,    1567],
        [1,    128,    2722])

    configs = [
        [1, 32, 32, [3, 3], 8, [3, 3], [1, 1], [1, 1, 1, 1]],
        [1, 32, 32, [4, 4], 8, [3, 3], [1, 1], [1, 1, 1, 1]],
        [1, 32, 32, [3, 5], 8, [3, 3], [1, 1], [1, 1, 1, 1]],
        [1, 32, 32, [5, 3], 8, [3, 3], [1, 1], [1, 1, 1, 1]],
        [1, 8, 8, [5, 5], 2, [3, 3], [1, 1], [1, 1, 1, 1]],
        [1, 128, 128, [56, 48], 32, [3, 3], [1, 1], [1, 1, 1, 1]],
        [1, 128, 128, [48, 56], 32, [3, 3], [1, 1], [1, 1, 1, 1]],
        [1, 64, 64, [3, 3], 8, [3, 3], [1, 1], [1, 1, 1, 1]],
        [1, 64, 64, [4, 4], 8, [3, 3], [1, 1], [1, 1, 1, 1]],
        [1, 64, 64, [3, 5], 8, [3, 3], [1, 1], [1, 1, 1, 1]],
        [1, 64, 64, [5, 3], 8, [3, 3], [1, 1], [1, 1, 1, 1]],
        [1, 16, 16, [5, 5], 2, [3, 3], [1, 1], [1, 1, 1, 1]],
        [1, 256, 256, [56, 48], 32, [3, 3], [1, 1], [1, 1, 1, 1]],
        [1, 256, 256, [48, 56], 32, [3, 3], [1, 1], [1, 1, 1, 1]],
        [1, 256, 256, [56, 56], 32, [3, 3], [1, 1], [1, 1, 1, 1]],
        [2, 256, 256, [56, 56], 32, [3, 3], [1, 1], [1, 1, 1, 1]]]

    for i in range(len(configs)):
        config = configs[i]
        test_fbgemm_conv_int8(config[0], config[1], config[2], config[3],
                              config[4], config[5], config[6], config[7])

    """
    if True:

	 shapes = (
		[4, 8, 2],
		[2, 16, 1],
		[4, 4, 2],
		[1, 8, 4],
		[16, 1, 1],
		[16, 2, 2],
		[8, 2, 4],
		[2, 2, 8])

	 values = (
		[1.0, 2.0, 0.0],
		[2.0, 2.0, 0.0],
		[3.0, 1.0, 0.0],
		[2.0, 3.0, 0.0],
		[1.0, 3.0, 0.0],
		[2.0, 3.0, 3.0],
		[2.0, 1.0, 2.0])

	 comb = []
	 for shape in shapes:
		for value in values:
			c = shape + value
			comb.append(c)

         for c in comb:
 	    test_fbgemm_packed_weights_with_requant_and_trans(c[0], c[1], c[2], c[3], c[4], c[5], True, True)
            test_fbgemm_packed_weights_with_requant_and_trans(c[0], c[1], c[2], c[3], c[4], c[5], True, False)
            test_fbgemm_packed_weights_with_requant_and_trans(c[0], c[1], c[2], c[3], c[4], c[5], False, True)
            test_fbgemm_packed_weights_with_requant_and_trans(c[0], c[1], c[2], c[3], c[4], c[5], False, False) 

    else:
    
         for shape in shapes_others:
              task = autotvm.task.create(
                  tune_fbgemm_packed_weights, args=(
                      shape[0] , shape[1] , shape[2] ), target='llvm')
              #print(task.config_space)
              print(len(task.config_space))
              # logging config (for printing tuning log to the screen)
              logging.getLogger('autotvm').setLevel(logging.DEBUG)
              logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

              measure_option = autotvm.measure_option(
                  builder='local',
                  runner=autotvm.LocalRunner(number=10, timeout=100000))

              tuner = autotvm.tuner.RandomTuner(task)
              log_file_name = "fbgemm_results_"+str(shape[0])+"_"+str(shape[1])+"_"+str(shape[2])+".log"
              tuner.tune(n_trial=150,
                         measure_option=measure_option,
                         callbacks=[autotvm.callback.log_to_file(log_file_name)])
    """
