import tvm
import numpy as np
from topi.x86.tensor_intrin import dot_16x1x16_int8_int8_int32



def test_fc_int8_acc32():
    n=1024
    k=1024
    m=1024

    X = tvm.placeholder((m, k), name='X', dtype="uint8")
    W = tvm.placeholder((n, k), name='W', dtype="int8")

    peak = 512/16*2*2*2
    print("Peak {} Gops/s".format(peak))
    memory_ops = n*k + m*k + 2*n*n
    gops_per_mm = 2*n*m*k

    def verify(target="llvm -mcpu=skylake-avx512"):
        if not tvm.module.enabled(target):
            print("skip because %s is not enabled..." % target)
            return

        ctx = tvm.context(target, 0)
        pc = dot_16x1x16_int8_int8_int32()
        ak = tvm.reduce_axis((0, k), name='k')
        packedW = tvm.placeholder((n/16, 16*(k/4), 4), name='packedW', dtype="int8")

        t_fc = tvm.compute((m, n), lambda i, j: tvm.sum(X[i, ak].astype("int32") * packedW[j/16, (ak/4)*16+j%16, ak%4].astype("int32"), axis=ak), name="F")
        t_sch = tvm.create_schedule(t_fc.op)
        a_x, a_y = t_fc.op.axis
        a_k, = t_fc.op.reduce_axis

        a_yo, a_yi = t_sch[t_fc].split(a_y, factor=16)
        a_xo, a_xi = t_sch[t_fc].split(a_x, factor=32)
        a_ko, a_ki = t_sch[t_fc].split(a_k, factor=4)
        a_koo, a_koi = t_sch[t_fc].split(a_ko, factor=4)
        t_sch[t_fc].reorder(a_yo, a_xo, a_xi, a_koo, a_koi, a_yi, a_ki)

        # measure multiple threading
        # t_sch[t_fc].parallel(a_xo)
        t_sch[t_fc].unroll(a_koi)
        t_sch[t_fc].tensorize(a_yi, pc)

        # print(tvm.lower(t_sch, [X, packedW, t_fc], simple_mode=True))

        t_func = tvm.build(t_sch, [X, packedW, t_fc], target="llvm -mcpu=skylake-avx512", name="intrinsic")
        t_evaluator = t_func.time_evaluator(t_func.entry_name, ctx, number=10)

        # generate the plain data
        a_ = np.random.uniform(1, 10, size=(m, k)).astype("uint8")
        b_ = np.random.uniform(1, 10,  size=(n, k)).astype("int8")

        packW = np.random.uniform(1, 10,  size=(n/16, 16*(k/4), 4)).astype("int8")
        # This occurs in pre_compute stage
        for r_idx in range(n/16):
            for s_idx in range(16*(k/4)):
                for t_idx in range(4):
            	    packW[r_idx][s_idx][t_idx] = b_[r_idx*16+s_idx%16][s_idx/16*4+t_idx]

        x = tvm.nd.array(a_, ctx)
        w = tvm.nd.array(packW, ctx)
        y = tvm.nd.array(np.zeros((m, n), dtype="int32"), ctx)
        result = t_evaluator(x, w, y)

        gops_per_sec = gops_per_mm/result.mean/1e9
        # verify the correctness
        tvm.testing.assert_allclose(y.asnumpy(), np.dot(a_, b_.T), rtol=0)
        print('Tensorization: running time: {:.3f} ms, {:.2f} Gops/s, effiency: {:.2f}'.format(result.mean*1000, gops_per_sec, gops_per_sec/peak))
            # print('TVM with x86 micro-kernel: %f' % result.mean)
        t_func.export_library("tensorize_acc32.o")

    verify()

if __name__ == "__main__":
    test_fc_int8_acc32()
