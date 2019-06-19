"""
A reference solution of convolution with requantization used to check
correctness of test_fbgemm_conv_int8(), which calls FBGEMM convolution
operator from TVM. This is called directly from the TVM python interface.

This is a python reimplementation of c++ reference solution from FBGEMM:
https://github.com/pytorch/FBGEMM/blob/master/src/RefImplementations.cc
"""

def conv_ref(MB, IC, OC, IN_DIM, OUT_DIM, G, K, stride, pad,
             A, A_zero_point, B):
    c_ref = [0 for i in range(MB * OUT_DIM[0] * OUT_DIM[1] * OC)]
    for n in range(MB):
        for h in range(OUT_DIM[0]):
            for w in range(OUT_DIM[1]):
                for g in range(G):
                    for m in range(OC / G):
                        sum = 0
                        for r in range(K[0]):
                            h_in = -pad[0] + h * stride[0] + r
                            for s in range(K[1]):
                                w_in = -pad[1] + w * stride[1] + s
                                for c in range(IC / G):
                                    a = 0
                                    if h_in < 0 or h_in >= IN_DIM[0] \
                                       or w_in < 0 or w_in >= IN_DIM[1]:
                                        a = A_zero_point
                                    else:
                                        a = A[((n * IN_DIM[0] + h_in)
                                            * IN_DIM[1] + w_in)
                                            * IC + g * (IC / G) + c]
                                    b = B[(((g * K[0] + r) * K[1] + s)
                                        * (IC / G) + c) * (OC / G) + m]
                                    sum += a * b;

                        c_ref[((n * OUT_DIM[0] + h)
                        * OUT_DIM[1] + w) * OC + g * (OC / G) + m] = sum;

    return c_ref


def im2col_ref(MB, IC, OC, IN_DIM, OUT_DIM, G, K, stride,
               pad, A, A_zero_point, length):
    Ao = [0 for i in range(length)]
    for n in range(MB):
        for h in range(OUT_DIM[0]):
            for w in range(OUT_DIM[1]):
                for r in range(K[0]):
                    h_in = -pad[0] + h * stride[0] + r
                    for s in range(K[1]):
                        w_in = -pad[1] + w * stride[1] + s
                        if h_in < 0 or h_in >= IN_DIM[0] \
                           or w_in < 0 or w_in >= IN_DIM[1]:
                            for g in range(G):
                                for c_ in range(IC / G):
                                    id = (((((n * OUT_DIM[0] + h)
                                    * OUT_DIM[1] + w) * G + g) * K[0] + r)
                                    * K[1] + s) * (IC / G)
                                    Ao[id + c_] = A_zero_point

                        else:
                            for g in range(G):
                                for c_ in range(IC / G):
                                    id = (((((n * OUT_DIM[0] + h)
                                    * OUT_DIM[1] + w) * G + g) * K[0] + r)
                                    * K[1] + s) * (IC / G)
                                    id_src = ((n * IN_DIM[0] + h_in)
                                    * IN_DIM[1] + w_in) * IC + g * (IC / G)
                                    Ao[id + c_] = A[id_src + c_]
    return Ao



def requantize_u8acc32_ref(M1, K1, ld1, Aint81, length1, KDimPerGroup1, G,
			               M, N, ld, inp, C_multiplier, C_zero_point,
                           A_zero_point, B_zero_point, col_offsets,
                           ncols_per_quant_group, NDim, NDim_OC):
    out = [0 for i in range(len(inp))]
    row_offsets = [0 for i in range(length1)]
    for g in range(G):
        for i in range(M1):
            sum = 0
            for k in range(K1):
                sum += Aint81[KDimPerGroup1 * g + i * ld1 + k]
            row_offsets[i] = sum

	for i in range(M):
            for j in range(N):
                raw = inp[NDim * g + i * ld + j]
                if A_zero_point:
                    raw -= A_zero_point * col_offsets[j + NDim * g]
                raw -= B_zero_point[j / ncols_per_quant_group + NDim_OC * g] \
                       * row_offsets[i]
                result = raw * C_multiplier[NDim_OC * g + j / ncols_per_quant_group]
                rounded = round(result) + C_zero_point
                out[NDim * g + i * ld + j] = max(0, min(255, rounded))
    return out

def col_offsets_with_zero_pt_s8acc32_ref(K, N, ld, OC, Bint8, B_zero_point,
                                         ncols_per_quant_group,
                                         G, col_lead, w_lead):
    col_offsets = [0 for i in range(OC)]
    for g in range(G):
        for j in range(N):
            total = 0
            for k in range(K):
                total += Bint8[g * w_lead + k * ld + j]
            col_offsets[g * col_lead + j] = total - \
            B_zero_point[j / ncols_per_quant_group] * K

    return col_offsets


def reference_solution(A, A_zero_point, W, MB, IC, OC, IN_DIM, OUT_DIM, G, K,
                       stride, pad, C_multiplier, B_zero_point, C_zero_point):
    Cint32_ref = conv_ref(MB, IC, OC, IN_DIM, OUT_DIM, G, K, stride, pad,
                 A, A_zero_point, W)


    im_in_dim = IN_DIM[0] * IN_DIM[1]
    kernel_dim = K[0] * K[1]
    im_out_dim = OUT_DIM[0] * OUT_DIM[1]

    MDim = MB * im_out_dim
    NDim = OC / G
    KDim = kernel_dim * IC
    KDimPerGroup = KDim / G

    OC_per_G = OC / G

    length_im2col = MDim * KDim
    A_im2col = im2col_ref(MB, IC, OC, IN_DIM, OUT_DIM, G, K,
                          stride, pad, A, A_zero_point, length_im2col)
    col_lead = OC_per_G
    w_lead = KDimPerGroup * OC_per_G
    col_offsets = \
    col_offsets_with_zero_pt_s8acc32_ref(KDimPerGroup, OC_per_G, OC_per_G, OC,
                                        W, B_zero_point, OC, G, col_lead, w_lead)

    NDim_OC = NDim / OC
    output = requantize_u8acc32_ref(MDim, KDimPerGroup, KDim, A_im2col, MDim,
                                    KDimPerGroup, G, MDim, NDim, G * NDim,
                                    Cint32_ref, C_multiplier, C_zero_point,
                                    A_zero_point, B_zero_point, col_offsets,
                                    OC, NDim, NDim_OC)

    return output
