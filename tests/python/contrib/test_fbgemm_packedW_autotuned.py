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


def test_fbgemm_packed_weights_with_requant(m, n, k, w_val, x_val, b_val, A_trans, W_trans):
    ctx = tvm.cpu(0)
    W = tvm.placeholder((k, n), name='W', dtype="uint8")
    w1 = np.random.uniform(w_val - 1, w_val + 2, size=(k, n)).astype(W.dtype)
    if W_trans:
        w = tvm.nd.array(w1.transpose(), ctx)
    else:
        w = tvm.nd.array(w1, ctx)
    my_packedw = tvm.get_global_func("tvm.contrib.fbgemm.pack_matrixB_int8")
    ww = my_packedw(w, 1)

    get_co_offsets = tvm.get_global_func("tvm.contrib.fbgemm.compute_col_offsets_int8")
    co = get_co_offsets(w, 1, 1)

    if A_trans:
        X = tvm.placeholder((k, m), name='X', dtype="int8")
    else:
        X = tvm.placeholder((m, k), name='X', dtype="int8")
    B = tvm.placeholder((n,), name='B', dtype="int")

    # quantization parameters will be got from Operator arguments
    X_qparams = QuantParams(scale=1.0, zero_point=0)
    W_qparams = QuantParams(scale=1.0, zero_point=0)
    Y_qparams = QuantParams(scale=1.0, zero_point=0)

    C = fbgemm.gemm_int8acc32_prepacked_with_requant(m, n, X, X_qparams, ww, W_qparams,
						     B, Y_qparams, co)
    #Y = tvm.compute((m, n), lambda i, j: C[i][j], name="Y")
    #s = tvm.create_schedule(Y.op)
    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [X, B, C], target="llvm", name="packedmatmul_with_requant")
    #print(tvm.lower(s, [X, B, C], simple_mode=True))
    f_evaluator = f.time_evaluator(f.entry_name, ctx, 10)
    x1 = np.random.uniform(x_val - 1, x_val + 2, size=(m, k)).astype(X.dtype)
    if A_trans:
        x = tvm.nd.array(x1.transpose(), ctx)
    else:
        x = tvm.nd.array(x1, ctx)
    b = tvm.nd.array(np.random.uniform(b_val, b_val + 2, size=(n,)).astype(B.dtype), ctx)
    y = tvm.nd.array(np.zeros((m, n), dtype=C.dtype), ctx)
    f(x,b,y)

    result = f_evaluator(x,b,y)
    print(result)
    gops_per_mm = 2*m*n*k
    gops_per_sec = gops_per_mm/result.mean/1e9
    print("M:{}, N:{}, K:{}".format(m,n,k))
    print(gops_per_sec)
    #print(y.asnumpy())
    #print(np.matmul(x1, w1) + b.asnumpy())

    tvm.testing.assert_allclose(
           y.asnumpy(), np.matmul(x1, w1) + b.asnumpy(), rtol=1e-5)


def test_fbgemm_conv_int8(MB, IC, OC, IN_DIM_lst, G, K_lst, stride_lst, pad_lst):
    ctx = tvm.cpu(0)
    spatial_dim = 2
    IN_DIM = tvm.nd.array(np.array(IN_DIM_lst).astype("int32"), ctx)
    K = tvm.nd.array(np.array(K_lst).astype("int32"), ctx)
    stride = tvm.nd.array(np.array(stride_lst).astype("int32"), ctx)
    pad = tvm.nd.array(np.array(pad_lst).astype("int32"), ctx)

    IN_DIMP = [0, 0]
    OUT_DIM = [0, 0]

    IN_DIMP[0] = IN_DIM_lst[0] + pad_lst[0] + pad_lst[2];
    OUT_DIM[0] = (IN_DIMP[0] - K_lst[0]) / stride_lst[0] + 1;

    IN_DIMP[1] = IN_DIM_lst[1] + pad_lst[1] + pad_lst[3];
    OUT_DIM[1] = (IN_DIMP[1] - K_lst[1]) / stride_lst[1] + 1;


    MDim = MB * OUT_DIM[0] * OUT_DIM[1];
    NDim = OC / G
    KDim = K_lst[0] * K_lst[1] * IC
    #print(MDim, NDim, KDim)
    # shapes
    input_shape = (MB, IN_DIM_lst[0], IN_DIM_lst[1], IC) #NHWC
    W_shape = (K_lst[0], K_lst[1], IC, OC / G) #RSCK
    Y_shape = (MB, OUT_DIM[0], OUT_DIM[1], OC) #NHWK
    # weight
    W = tvm.placeholder(W_shape, name='W', dtype="int8")
    wa_length = K_lst[0] * K_lst[1] * IC * OC / G
    wa = [random.randint(-4, 4) for i in range(wa_length)]
    w = tvm.nd.array(np.reshape(np.array(wa), W_shape).astype(W.dtype), ctx)

    # packing of weight
    my_packedw = tvm.get_global_func("tvm.contrib.fbgemm.pack_matrixB_int8_conv")

    ww = my_packedw(w, MB, IC, OC, IN_DIM_lst[0], IN_DIM_lst[1], G, K_lst[0], K_lst[1],
                    stride_lst[0], stride_lst[1], pad_lst[0], pad_lst[1], pad_lst[2], pad_lst[3])

    # input (X)
    X = tvm.placeholder(input_shape, name='X', dtype="uint8")

    # quantization parameters will be got from Operator arguments
    X_zero_point = 4
    W_zero_point = -2
    Y_zero_point = 5

    # column offset
    get_co_offsets = \
    tvm.get_global_func("tvm.contrib.fbgemm.compute_col_offsets_int8_conv")
    co = get_co_offsets(w, W_zero_point,
                        MB, IC, OC, IN_DIM_lst[0], IN_DIM_lst[1], G, K_lst[0], K_lst[1],
                    stride_lst[0], stride_lst[1], pad_lst[0], pad_lst[1], pad_lst[2], pad_lst[3])

    C_multiplier = 0.0878014

    C = fbgemm.conv_int8(Y_shape, X, X_zero_point, ww, W,
                         W_zero_point, Y_zero_point, C_multiplier, co,
                         MB, IC, OC, IN_DIM_lst, G, K_lst, stride_lst, pad_lst)

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [X, W, C], target="llvm", name="conv_int8")

    x_length = MB * IN_DIM_lst[0] * IN_DIM_lst[1] * IC
    xa = [random.randint(0, 5) for i in range(x_length)]
    x = tvm.nd.array(np.reshape(np.array(xa), input_shape).astype(X.dtype), ctx)
    y = tvm.nd.array(np.zeros(Y_shape, dtype=C.dtype), ctx)

    f_evaluator = f.time_evaluator(f.entry_name, ctx, 50)
    for i in range(3):
        f(x, w, y)
    result = f_evaluator(x,w,y)
    gops_per_mm = 2 * KDim * MDim * NDim
    gops_per_sec = gops_per_mm/result.mean/1e9
    print(gops_per_sec)
    #y_ref = reference_solution(xa, X_zero_point, wa, MB, IC, OC, IN_DIM_lst,
     #                          OUT_DIM, G, K_lst, stride_lst, pad_lst, [C_multiplier],
      #                         [W_zero_point], Y_zero_point)
    #y_ref = np.reshape(np.array(y_ref), Y_shape)
    #tvm.testing.assert_allclose(y.asnumpy(), y_ref, rtol=1e-5)


def isValidConfig(mcb, ncb, kcb, mr, nr, nr_min, row_interleave):
    if (mcb % mr):
        return False
    if (ncb % nr):
        return False
    if mr * nr / nr_min > 28:
        return False
    return True

@autotvm.template
def test_fbgemm_conv_int8_autotuned(MB, IC, OC, IN_DIM_lst, G, K_lst, stride_lst, pad_lst):
    ROW_INTERLEAVE = 4
    """
    MCBs = [48, 98, 144, 192, 240, 140]
    NCBs = [16, 32, 64, 128, 48, 98, 192, 384]
    KCBs = [256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 960, 1024]
    MRs = [1, 2, 4, 8, 3, 6, 12, 24, 28]
    NRs = [16, 32]
    NR_MINs = [16]
    """
    valid_configs = []
    """
    for mcb in MCBs:
        for ncb in NCBs:
            for kcb in KCBs:
                for mr in MRs:
                    for nr in NRs:
                        #if (isValidConfig(mcb, ncb, kcb, mr, nr, 16, 4)):
		        valid_configs.append((mcb, ncb, kcb, mr, nr, 16, 4))
    """
    #adding the default search point
    valid_configs.append((56, 32, 256, 14, 32, 16, 4))
    configs = autotvm.get_config()
    validate_func = lambda conf: isValidConfig(conf[0], conf[1], conf[2], conf[3], conf[4], conf[5], conf[6])
    configs.define_knob("VAL_CNFG", valid_configs, validate_func=validate_func)


    ctx = tvm.cpu(0)
    spatial_dim = 2
    IN_DIM = tvm.nd.array(np.array(IN_DIM_lst).astype("int32"), ctx)
    K = tvm.nd.array(np.array(K_lst).astype("int32"), ctx)
    stride = tvm.nd.array(np.array(stride_lst).astype("int32"), ctx)
    pad = tvm.nd.array(np.array(pad_lst).astype("int32"), ctx)

    IN_DIMP = [0, 0]
    OUT_DIM = [0, 0]

    IN_DIMP[0] = IN_DIM_lst[0] + pad_lst[0] + pad_lst[2];
    OUT_DIM[0] = (IN_DIMP[0] - K_lst[0]) / stride_lst[0] + 1;

    IN_DIMP[1] = IN_DIM_lst[1] + pad_lst[1] + pad_lst[3];
    OUT_DIM[1] = (IN_DIMP[1] - K_lst[1]) / stride_lst[1] + 1;

    MDim = MB * OUT_DIM[0] * OUT_DIM[1];
    NDim = OC / G
    KDim = K_lst[0] * K_lst[1] * IC
    no_ops = 2 * MDim * NDim * KDim
    configs.add_flop(no_ops)

    # shapes
    input_shape = (MB, IN_DIM_lst[0], IN_DIM_lst[1], IC) #NHWC
    W_shape = (K_lst[0], K_lst[1], IC, OC / G) #RSCK
    Y_shape = (MB, OUT_DIM[0], OUT_DIM[1], OC) #NHWK
    # weight
    W = tvm.placeholder(W_shape, name='W', dtype="int8")
    wa_length = K_lst[0] * K_lst[1] * IC * OC / G
    wa = [random.randint(-4, 4) for i in range(wa_length)]
    w = tvm.nd.array(np.reshape(np.array(wa), W_shape).astype(W.dtype), ctx)

    # packing of weight
    my_packedw = tvm.get_global_func("tvm.contrib.fbgemm.pack_matrixB_int8_conv")

    ww = my_packedw(w, MB, IC, OC, IN_DIM_lst[0], IN_DIM_lst[1], G, K_lst[0], K_lst[1],
		    stride_lst[0], stride_lst[1], pad_lst[0], pad_lst[1], pad_lst[2], pad_lst[3],
                    configs["VAL_CNFG"].val[0],
                    configs["VAL_CNFG"].val[1],
                    configs["VAL_CNFG"].val[2],
                    configs["VAL_CNFG"].val[3],
                    configs["VAL_CNFG"].val[4],
                    configs["VAL_CNFG"].val[5],
    		        ROW_INTERLEAVE)

    # input (X)
    X = tvm.placeholder(input_shape, name='X', dtype="uint8")

    # quantization parameters will be got from Operator arguments
    X_zero_point = 4
    W_zero_point = -2
    Y_zero_point = 5

    # column offset
    get_co_offsets = \
    tvm.get_global_func("tvm.contrib.fbgemm.compute_col_offsets_int8_conv")
    co = get_co_offsets(w, W_zero_point,
                        MB, IC, OC, IN_DIM_lst[0], IN_DIM_lst[1], G, K_lst[0], K_lst[1],
                        stride_lst[0], stride_lst[1], pad_lst[0], pad_lst[1], pad_lst[2], pad_lst[3])

    C_multiplier = 0.0878014

    C = fbgemm.conv_int8(Y_shape, X, X_zero_point, ww, W,
                         W_zero_point, Y_zero_point, C_multiplier, co,
                         MB, IC, OC, IN_DIM_lst, G, K_lst, stride_lst, pad_lst,
                         1, True,
                    	 configs["VAL_CNFG"].val[0],
                    	 configs["VAL_CNFG"].val[1],
                    	 configs["VAL_CNFG"].val[2],
                    	 configs["VAL_CNFG"].val[3],
                    	 configs["VAL_CNFG"].val[4],
                    	 configs["VAL_CNFG"].val[5],
                         ROW_INTERLEAVE)


    s = tvm.create_schedule(C.op)
    return s, [X, W, C]


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


    if False:
      im2col_configs = [
            [1, 64, 256, [56, 56], 1, [1, 1], [1, 1], [0, 0, 0, 0]],
             [1, 128, 512, [28, 28], 1, [1, 1], [1, 1], [0, 0, 0, 0]],
             [1, 256, 512, [56, 56], 1, [1, 1], [2, 2], [0, 0, 0, 0]],
             [1, 256, 1024, [14, 14], 1, [1, 1], [1, 1], [0, 0, 0, 0]],
             [1, 512, 1024, [28, 28], 1, [1, 1], [2, 2], [0, 0, 0, 0]],
             [1, 512, 2048, [7, 7], 1, [1, 1], [1, 1], [0, 0, 0, 0]],
             [1, 1024, 2048, [14, 14], 1, [1, 1], [2, 2], [0, 0, 0, 0]],
             [1, 3, 64, [224, 224], 1, [7, 7], [2, 2], [3, 3, 3, 3]],
             [1, 64, 64, [56, 56], 1, [1, 1], [1, 1], [0, 0, 0, 0]],
             [1, 64, 64, [56, 56], 1, [3, 3], [1, 1], [1, 1, 1, 1]],
             [1, 256, 64, [56, 56], 1, [1, 1], [1, 1], [0, 0, 0, 0]],
             [1, 256, 128, [56, 56], 1, [1, 1], [1, 1], [0, 0, 0, 0]],
             [1, 128, 128, [56, 56], 1, [3, 3], [2, 2], [1, 1, 1, 1]],
             [1, 512, 128, [28, 28], 1, [1, 1], [1, 1], [0, 0, 0, 0]],
             [1, 128, 128, [28, 28], 1, [3, 3], [1, 1], [1, 1, 1, 1]],
             [1, 512, 256, [28, 28], 1, [1, 1], [1, 1], [0, 0, 0, 0]],
             [1, 256, 256, [28, 28], 1, [3, 3], [2, 2], [1, 1, 1, 1]],
             [1, 1024, 256, [14, 14], 1, [1, 1], [1, 1], [0, 0, 0, 0]],
             [1, 256, 256, [14, 14], 1, [3, 3], [1, 1], [1, 1, 1, 1]],
            [1, 1024, 512, [14, 14], 1, [1, 1], [1, 1], [0, 0, 0, 0]],
            [1, 512, 512, [14, 14], 1, [3, 3], [2, 2], [1, 1, 1, 1]],
            [1, 2048, 512, [7, 7], 1, [1, 1], [1, 1], [0, 0, 0, 0]],
            [1, 512, 512, [7, 7], 1, [3, 3], [1, 1], [1, 1, 1, 1]]
              ]
      for config in im2col_configs:
          test_fbgemm_conv_int8(config[0], config[1], config[2], config[3],
                                config[4], config[5], config[6], config[7])
         #fbgemm_packed_weights(1024, 1024, 1024)
         #for shape in shapes_others:
              #test_fbgemm_packed_weights_with_requant(shape[0], shape[1], shape[2], 0, 0, 0, False, False)
    else:
      im2col_configs = [

             [1, 64, 256, [56, 56], 1, [1, 1], [1, 1], [0, 0, 0, 0]],
             [1, 128, 512, [28, 28], 1, [1, 1], [1, 1], [0, 0, 0, 0]],
             [1, 256, 512, [56, 56], 1, [1, 1], [2, 2], [0, 0, 0, 0]],
             [1, 256, 1024, [14, 14], 1, [1, 1], [1, 1], [0, 0, 0, 0]],
             [1, 512, 1024, [28, 28], 1, [1, 1], [2, 2], [0, 0, 0, 0]],
             [1, 512, 2048, [7, 7], 1, [1, 1], [1, 1], [0, 0, 0, 0]],
             [1, 1024, 2048, [14, 14], 1, [1, 1], [2, 2], [0, 0, 0, 0]],
             [1, 3, 64, [224, 224], 1, [7, 7], [2, 2], [3, 3, 3, 3]],
             [1, 64, 64, [56, 56], 1, [1, 1], [1, 1], [0, 0, 0, 0]],
             [1, 64, 64, [56, 56], 1, [3, 3], [1, 1], [1, 1, 1, 1]],
             [1, 256, 64, [56, 56], 1, [1, 1], [1, 1], [0, 0, 0, 0]],
             [1, 256, 128, [56, 56], 1, [1, 1], [1, 1], [0, 0, 0, 0]],
             [1, 128, 128, [56, 56], 1, [3, 3], [2, 2], [1, 1, 1, 1]],
             [1, 512, 128, [28, 28], 1, [1, 1], [1, 1], [0, 0, 0, 0]],
             [1, 128, 128, [28, 28], 1, [3, 3], [1, 1], [1, 1, 1, 1]],
             [1, 512, 256, [28, 28], 1, [1, 1], [1, 1], [0, 0, 0, 0]],
             [1, 256, 256, [28, 28], 1, [3, 3], [2, 2], [1, 1, 1, 1]],
             [1, 1024, 256, [14, 14], 1, [1, 1], [1, 1], [0, 0, 0, 0]],
             [1, 256, 256, [14, 14], 1, [3, 3], [1, 1], [1, 1, 1, 1]],
            [1, 1024, 512, [14, 14], 1, [1, 1], [1, 1], [0, 0, 0, 0]],
            [1, 512, 512, [14, 14], 1, [3, 3], [2, 2], [1, 1, 1, 1]],
            [1, 2048, 512, [7, 7], 1, [1, 1], [1, 1], [0, 0, 0, 0]],
            [1, 512, 512, [7, 7], 1, [3, 3], [1, 1], [1, 1, 1, 1]]

              ]

      configs = [#[1, 32, 32, [14, 14], 1, [3, 3], [1, 1], [0, 0, 0, 0]],
        #[1, 32, 16, [12, 14], 4, [3, 3], [1, 1], [0, 0, 0, 0]],
        [1, 8, 8, [4, 4], 1, [3, 3], [1, 1], [1, 1, 0, 0]],
        [1, 32, 32, [14, 14], 1, [3, 3], [1, 1], [0, 0, 0, 0]],
        [1, 32, 32, [14, 14], 1, [3, 3], [1, 1], [1, 1, 1, 1]],
        [2, 32, 32, [14, 14], 1, [3, 3], [1, 1], [0, 0, 0, 0]],
        [2, 32, 32, [28, 14], 1, [3, 3], [1, 1], [1, 1, 0, 0]],
        [1, 32, 16, [12, 14], 4, [3, 3], [1, 1], [0, 0, 0, 0]],
        [2, 32, 16, [16, 14], 4, [3, 3], [1, 1], [0, 0, 0, 0]],
        [1, 544, 544, [14, 14], 1, [3, 3], [2, 2], [1, 1, 1, 1]],
        [1, 8, 8, [4, 4], 1, [3, 3], [1, 1], [1, 1, 0, 0]],
        [1, 3, 64, [224, 224], 1, [7, 7], [2, 2], [3, 3, 3, 3]],
        [1, 256, 256, [56, 56], 32, [3, 3], [2, 2], [1, 1, 1, 1]],
        [1, 512, 512, [28, 28], 32, [3, 3], [2, 2], [1, 1, 1, 1]],
        [1, 1024, 1024, [14, 14], 32, [3, 3], [2, 2], [1, 1, 1, 1]],
        [1, 1024, 1024, [7, 7], 32, [3, 3], [1, 1], [1, 1, 1, 1]],
        [50, 3, 64, [224, 224], 1, [7, 7], [2, 2], [3, 3, 3, 3]],
        [50, 256, 256, [56, 56], 32, [3, 3], [2, 2], [1, 1, 1, 1]],
        [50, 512, 512, [28, 28], 32, [3, 3], [2, 2], [1, 1, 1, 1]],
        [50, 1024, 1024, [14, 14], 32, [3, 3], [2, 2], [1, 1, 1, 1]],
        [50, 1024, 1024, [7, 7], 32, [3, 3], [1, 1], [1, 1, 1, 1]]
        ]

      for config in im2col_configs:
          task = autotvm.task.create(
                 test_fbgemm_conv_int8_autotuned,
                 args=(config[0], config[1], config[2], config[3],
                 config[4], config[5], config[6], config[7]), target='llvm')
          #print(task.config_space)
          print(len(task.config_space))
          # logging config (for printing tuning log to the screen)
          logging.getLogger('autotvm').setLevel(logging.DEBUG)
          logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))
          measure_option = autotvm.measure_option(
              builder='local',
              runner=autotvm.LocalRunner(number=100, timeout=100000))
          tuner = autotvm.tuner.RandomTuner(task)
          name = str(config)
          log_file_name = "fbgemm_n_default_200_results_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.log"\
	  .format(config[0], config[1], config[2], config[3][0], config[3][1],
                 config[4], config[5][0], config[5][1], config[6][0], config[6][1],
		 config[7][0], config[7][1], config[7][2], config[7][3])
          tuner.tune(n_trial=200,
                     measure_option=measure_option,
                     callbacks=[autotvm.callback.log_to_file(log_file_name)])
    """   
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
