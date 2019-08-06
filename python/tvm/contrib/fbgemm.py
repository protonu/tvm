"""External function interface to FBGEMM libraries."""
from __future__ import absolute_import as _abs

from .. import api as _api
from .. import intrin as _intrin
from .._ffi.function import _init_api


def matmul_fp16(A, B, nthreads=1):
    """Create an extern op that compute matrix multiply with fbgemm fp16.

    Parameters
    ----------
    A : Tensor
        2D array M*K
    B : Tensor
        2D array K*N

    Returns
    -------
    C : Tensor
        2D array out M*N
    """
    n = B.shape[1]
    m = A.shape[0]
    return _api.extern(
        (m, n), [A, B],
        lambda ins, outs: _intrin.call_packed(
            "tvm.contrib.fbgemm.matmul_fp16",
            ins[0], ins[1], outs[0], nthreads), name="C")

def packB_with_alloacted_tensor(num_rows, num_cols, X, nthreads=1,
                         	autotune = False, MCB = 56, NCB = 32, KCB = 256, 
                                MR = 14, NR = 32, NR_MIN = 16, ROW_INTERLEAVE = 4):
    return _api.extern(
        (num_rows, num_cols), [X],
            lambda ins, outs: _intrin.call_packed(
              	    "tvm.contrib.fbgemm.packB_with_allocated_tensor",
                    ins[0], outs[0], 
                    MCB, NCB, KCB, MR, NR, NR_MIN, ROW_INTERLEAVE), 
                    name="PB", dtype="int8")           

def packedgemm_U8S8ACC32(m,n, A, PackedB,  nthreads=1,
                         	autotune = False, MCB = 56, NCB = 32, KCB = 256, 
                                MR = 14, NR = 32, NR_MIN = 16, ROW_INTERLEAVE = 4):
    return _api.extern(
        (m, n), [A, PackedB],
            lambda ins, outs: _intrin.call_packed(
              	    "tvm.contrib.fbgemm.packedgemm_U8S8ACC32",
                    ins[0], ins[1], outs[0], 
                    MCB, NCB, KCB, MR, NR, NR_MIN, ROW_INTERLEAVE), 
                    name="C", dtype="int")           



def gemm_int8acc32_prepacked(m, n, X, X_qparams, packedW, W_qparams,
                                B, Y_qparams, col_offsets, nthreads=1,
                         	autotune = False, MCB = 56, NCB = 32, KCB = 256, 
                                MR = 14, NR = 32, NR_MIN = 16, ROW_INTERLEAVE = 4):

    ReQuant_multiplier=X_qparams.scale * W_qparams.scale / Y_qparams.scale
    if autotune:
         return _api.extern(
                 (m, n), [X, B],
                 lambda ins, outs: _intrin.call_packed(
             	    "tvm.contrib.fbgemm.gemmint8acc32packedwt",
                    ins[0], packedW, ins[1], outs[0], X_qparams.zero_point, 
                    W_qparams.zero_point, Y_qparams.zero_point, 
                    ReQuant_multiplier, col_offsets, nthreads, 
                    MCB, NCB, KCB, MR, NR, NR_MIN, ROW_INTERLEAVE), 
                    name="C", dtype="int")           

    else:
         return _api.extern(
                 (m, n), [X, B],
                 lambda ins, outs: _intrin.call_packed(
             	    "tvm.contrib.fbgemm.gemmint8acc32packedwt",
                    ins[0], packedW, ins[1], outs[0], X_qparams.zero_point, 
                    W_qparams.zero_point, Y_qparams.zero_point, 
                    ReQuant_multiplier, col_offsets, nthreads), name="C", dtype="int")           

def gemm_int8acc32_prepacked_for_tuning(m, n, W, X, X_qparams, packedW, W_qparams,
                                B, Y_qparams, col_offsets, nthreads=1,
                         	autotune = False, MCB = 56, NCB = 32, KCB = 256, MR = 14, NR = 32, NR_MIN = 16, ROW_INTERLEAVE = 4):

    ReQuant_multiplier=X_qparams.scale * W_qparams.scale / Y_qparams.scale

    if autotune:
         return _api.extern(
                 (m, n), [X, W, B],
                 lambda ins, outs: _intrin.call_packed(
             	    "tvm.contrib.fbgemm.gemmint8acc32packedwt_for_tuning",
                    ins[0], ins[1], ins[2], outs[0], X_qparams.zero_point, 
                    W_qparams.zero_point, Y_qparams.zero_point, 
                    ReQuant_multiplier, col_offsets, nthreads, 
                    MCB, NCB, KCB, MR, NR, NR_MIN, ROW_INTERLEAVE), 
                    name="C", dtype="int")           

    else:
         return _api.extern(
                 (m, n), [X, W, B],
                 lambda ins, outs: _intrin.call_packed(
             	    "tvm.contrib.fbgemm.gemmint8acc32packedwt_for_tuning",
                    ins[0], ins[1], ins[2], outs[0], X_qparams.zero_point, 
                    W_qparams.zero_point, Y_qparams.zero_point, 
                    ReQuant_multiplier, col_offsets, nthreads), name="C", dtype="int")           

def fully_connected_int8(X, X_qparams, W, W_qparams, B, Y_qparams, nthreads=1,
                         autotune=False, MCB=56, NCB=32, KCB=256, MR=14, NR=32, NR_MIN=16, ROW_INTERLEAVE=4):
    m = X.shape[0]
    n = W.shape[0]
    ReQuant_multiplier = X_qparams.scale * W_qparams.scale / Y_qparams.scale

    if autotune:
        return _api.extern(
            (m, n), [X, W, B],
            lambda ins, outs: _intrin.call_packed(
                "tvm.contrib.fbgemm.fully_connected_int8",
                ins[0], ins[1], ins[2], outs[0], X_qparams.zero_point, W_qparams.zero_point, Y_qparams.zero_point, ReQuant_multiplier, nthreads, MCB, NCB, KCB, MR, NR, NR_MIN, ROW_INTERLEAVE), name="C", dtype="int8")
    else:
        return _api.extern(
            (m, n), [X, W, B],
            lambda ins, outs: _intrin.call_packed(
                "tvm.contrib.fbgemm.fully_connected_int8",
                ins[0], ins[1], ins[2], outs[0], X_qparams.zero_point, W_qparams.zero_point, Y_qparams.zero_point, ReQuant_multiplier, nthreads), name="C", dtype="int8")


def gemm_int8acc32_prepacked_with_requant(m, n, X, X_qparams, packedW, W_qparams,
                                B, Y_qparams, col_offsets, 
				nthreads=1,
                         	autotune = False, MCB = 56, NCB = 32, KCB = 256,
                                MR = 14, NR = 32, NR_MIN = 16, ROW_INTERLEAVE = 4):

    ReQuant_multiplier=X_qparams.scale * W_qparams.scale / Y_qparams.scale
    if autotune:
         return _api.extern(
                 (m, n), [X, B],
                 lambda ins, outs: _intrin.call_packed(
             	    "tvm.contrib.fbgemm.gemmint8acc32packedwt_with_requant",
                    ins[0], packedW, ins[1], outs[0], X_qparams.zero_point,
                    W_qparams.zero_point, Y_qparams.zero_point,
                    ReQuant_multiplier, col_offsets, False,
		    nthreads,
                    MCB, NCB, KCB, MR, NR, NR_MIN, ROW_INTERLEAVE),
                    name="C", dtype="int8")
    else:
         return _api.extern(
                 (m, n), [X, B],
                 lambda ins, outs: _intrin.call_packed(
             	    "tvm.contrib.fbgemm.gemmint8acc32packedwt_with_requant",
                    ins[0], packedW, ins[1], outs[0], X_qparams.zero_point,
                    W_qparams.zero_point, Y_qparams.zero_point,
                    ReQuant_multiplier, col_offsets, False,
		    nthreads), name="C", dtype="int8")

def conv_int8(Y_shape, X, X_zero_point, W, Wmat, W_zero_point,
              Y_zero_point, C_multiplier, column_offset,
			  MB, IC, OC, IN_DIM, G, K, stride, pad, nthreads=1,
              autotune = False, MCB = 144, NCB = 16, KCB = 320,
              MR = 6, NR = 16, NR_MIN = 16, ROW_INTERLEAVE = 4):
#(144, 16, 320, 6, 16, 16, 4)
    if autotune:
	 print(MCB, NCB, KCB, MR, NR, NR_MIN, ROW_INTERLEAVE)
         return _api.extern(
             Y_shape, [X, Wmat],
             lambda ins, outs: _intrin.call_packed(
         	    "tvm.contrib.fbgemm.conv_int8",
                ins[0], W, outs[0], X_zero_point, W_zero_point,
                Y_zero_point,
                C_multiplier,
                column_offset, ins[1],
                MB, IC, OC, IN_DIM[0], IN_DIM[1], G, K[0], K[1],
		stride[0], stride[1], pad[0], pad[1], pad[2], pad[3],
                nthreads,
                MCB, NCB, KCB, MR, NR, NR_MIN, ROW_INTERLEAVE),
                name="C", dtype="uint8")
    else:
         return _api.extern(
             Y_shape, [X, Wmat],
             lambda ins, outs: _intrin.call_packed(
         	    "tvm.contrib.fbgemm.conv_int8",
                ins[0], W, outs[0], X_zero_point, W_zero_point,
                Y_zero_point,
                C_multiplier,
                column_offset, ins[1],
                MB, IC, OC, IN_DIM[0], IN_DIM[1], G, K[0], K[1],
                stride[0], stride[1], pad[0], pad[1], pad[2], pad[3],
	        nthreads), name="C", dtype="uint8")


_init_api("tvm.contrib.fbgemm")
