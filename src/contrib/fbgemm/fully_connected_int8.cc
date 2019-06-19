/*!
 *  Copyright (c) 2019 by Contributors
 * \file Use external fbgemm library call.
 */

#include <cpuinfo.h>
#include <dmlc/logging.h>
#include <fbgemm/Fbgemm.h>
#include <fbgemm/FbgemmFP16.h>
#include <fbgemm/QuantUtilsAvx2.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/util.h>
#include <memory>
#include <random>
#include "fbgemm_utils.h"

#include <sys/types.h>
#include <unistd.h>
#include <chrono>

namespace tvm {
namespace runtime {
using namespace fbgemm;

using packbmatrix = PackBMatrix<std::int8_t, std::int32_t>;
using packweight = PackWeightsForConv<2>;

template <>
struct extension_class_info<packbmatrix> {
  static const int code = 19;
};

template <>
struct extension_class_info<packweight> {
  static const int code = 20;
};

TVM_REGISTER_EXT_TYPE(packbmatrix);
TVM_REGISTER_EXT_TYPE(packweight);
}  // namespace runtime
}  // namespace tvm

namespace tvm {
namespace contrib {

using namespace runtime;
using namespace fbgemm;

TVM_REGISTER_GLOBAL("tvm.contrib.fbgemm.print_packb")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      void* pck_b = args[0];
      packbmatrix* B =
          reinterpret_cast<PackBMatrix<std::int8_t, std::int32_t>*>(pck_b);
      B->printPackedMatrix("B");
    });

TVM_REGISTER_GLOBAL("tvm.contrib.fbgemm.print_col_offsets")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      void* co = args[0];
      std::vector<std::int32_t>* coffsts =
          reinterpret_cast<std::vector<std::int32_t>*>(co);
      std::cout << "size of col offsets" << coffsts->size() << std::endl;
    });

//Add different implementation for transposed and untransposed weight.
TVM_REGISTER_GLOBAL("tvm.contrib.fbgemm.pack_matrixB_int8")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      bool trans = args[2];
      matrix_op_t trans_params = matrix_op_t::Transpose;
      DLTensor* W = args[0];
      int threads = args[1];

      CHECK_EQ(W->ndim, 2);

      int k = W->shape[0]; 
      int n = W->shape[1];
      int ld = n;
      if (trans) {
        trans_params = matrix_op_t::NoTranspose;
	ld = k;
      }
      BlockingFactors params;
      if (args.size() > 3) {
        int cntr = 3;
        params.MCB = args[cntr];
        params.NCB = args[cntr + 1]; 
        params.KCB = args[cntr + 2];
        params.MR = args[cntr + 3];
        params.NR = args[cntr + 4];
        params.NR_MIN = args[cntr + 5];
        params.ROW_INTERLEAVE = args[cntr + 6];

        auto packB = new PackBMatrix<std::int8_t, std::int32_t>(
            trans_params, k, n,
            reinterpret_cast<const std::int8_t*>(W->data), ld, nullptr, 1,
            &params);
        *ret = packB;

      } else {
        auto packB = new PackBMatrix<std::int8_t, std::int32_t>(
            trans_params, k, n,
            reinterpret_cast<const std::int8_t*>(W->data), ld, nullptr, 1);
        *ret = packB;
      }

    });

TVM_REGISTER_GLOBAL("tvm.contrib.fbgemm.pack_matrixB_int8_conv")
    .set_body([](TVMArgs args, TVMRetValue* ret) {

    DLTensor* W = args[0];

    int spatial_dim = args[1];
    int cntr = 2;
    int MB = args[cntr];
    int IC = args[cntr + 1];
    int OC = args[cntr + 2];

    DLTensor* id_addr = args[cntr + 3];
    int* id_pr = reinterpret_cast<int*>(id_addr->data);
    std::array<int, 2> IN_DIM = {0, 0};
    IN_DIM[0] = id_pr[0];
    IN_DIM[1] = id_pr[1];

    int G = args[cntr + 4];

    DLTensor* k_addr = args[cntr + 5];
    int* k_pr = reinterpret_cast<int*>(k_addr->data);
    std::array<int, 2> K = {0, 0};
    K[0] = k_pr[0];
    K[1] = k_pr[1];

    DLTensor* s_addr = args[cntr + 6];
    int* s_pr = reinterpret_cast<int*>(s_addr->data);
    std::array<int, 2> stride = {0, 0};
    stride[0] = s_pr[0];
    stride[1] = s_pr[1];

    DLTensor* pad_addr = args[cntr + 7];
    int* p_pr = reinterpret_cast<int*>(pad_addr->data);
    std::array<int, 4> pad = {0, 0, 0, 0};
    pad[0] = p_pr[0];
    pad[1] = p_pr[1];
    pad[2] = p_pr[2];
    pad[3] = p_pr[3];

    conv_param_t<> conv_p = conv_param_t<>(MB, IC, OC, IN_DIM, G, K, stride, pad);

    BlockingFactors params;

    if (args.size() > 11) {
      int cntr = 10;
      params.MCB = args[cntr];
      params.NCB = args[cntr + 1];
      params.KCB = args[cntr + 2];
      params.MR = args[cntr + 3];
      params.NR = args[cntr + 4];
      params.NR_MIN = args[cntr + 5];
      params.ROW_INTERLEAVE = args[cntr + 6];
      PackWeightsForConv<2>* packedB =
      new PackWeightsForConv<2>(conv_p, reinterpret_cast<std::int8_t*>(W->data));
      *ret = packedB;
      
    } else {
      PackWeightsForConv<2>* packedB =
      new PackWeightsForConv<2>(conv_p, reinterpret_cast<std::int8_t*>(W->data));
      *ret = packedB;
    } 
    
});  

//Add different implementation for transposed and untransposed weight.
TVM_REGISTER_GLOBAL("tvm.contrib.fbgemm.compute_col_offsets_int8")
    .set_body([](TVMArgs args, TVMRetValue* ret) {

      bool trans = args[3];

      DLTensor* W = args[0];
      int threads = args[1];
      std::int32_t w_zero_point = args[2];

      int k = W->shape[0];
      int n = W->shape[1];

      if (trans) { // N * K; transposed
        int inter = k;
        k = n;
        n = inter;
      }

      std::vector<TensorQuantizationParams> temp_qparams;
      temp_qparams.push_back(TensorQuantizationParams{1.0, w_zero_point});

      std::vector<std::int32_t>* column_offsets_ =
          new std::vector<std::int32_t>;
      ComputeColumnOffsets<std::int8_t>(
          k, n, reinterpret_cast<const std::int8_t*>(W->data), temp_qparams,
          *column_offsets_);
      *ret = column_offsets_;

    });

void col_offsets_with_zero_pt_s8acc32(
   int K,
   int N,
   int ld,
   const int8_t* Bint8,
   const int32_t* B_zero_point,
   int32_t* col_offsets,
   int ncols_per_quant_group) {
       
     for (int j = 0; j < N; ++j) {
       int32_t sum = 0;
       for (int k = 0; k < K; ++k) {
         sum += Bint8[k * ld + j];
       }
       col_offsets[j] = sum - B_zero_point[j / ncols_per_quant_group] * K;
     }
 } 

TVM_REGISTER_GLOBAL("tvm.contrib.fbgemm.compute_col_offsets_int8_conv")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
    
     // ARGUMENTS
    DLTensor* B = args[0]; // the weight
    int32_t Bzp = args[1];
      std::vector<int32_t> Bint8_zero_point = {Bzp};
    
     // conv_p
    int cntr = 3;
    int MB = args[cntr];
    int IC = args[cntr + 1];
    int OC = args[cntr + 2];
    DLTensor* id_addr = args[cntr + 3];
    int* id_pr = reinterpret_cast<int*>(id_addr->data);
    std::array<int, 2> IN_DIM = {0, 0};
    IN_DIM[0] = id_pr[0];
    IN_DIM[1] = id_pr[1];
    int G = args[cntr + 4];
    DLTensor* k_addr = args[cntr + 5];
    int* k_pr = reinterpret_cast<int*>(k_addr->data);
    std::array<int, 2> K = {0, 0};
    K[0] = k_pr[0];
    K[1] = k_pr[1];
    DLTensor* s_addr = args[cntr + 6];
    int* s_pr = reinterpret_cast<int*>(s_addr->data);
    std::array<int, 2> stride = {0, 0};
    stride[0] = s_pr[0];
    stride[1] = s_pr[1];
    
    DLTensor* pad_addr = args[cntr + 7];
    int* p_pr = reinterpret_cast<int*>(pad_addr->data);
    std::array<int, 4> pad = {0, 0, 0, 0};
    pad[0] = p_pr[0];
    pad[1] = p_pr[1];
    pad[2] = p_pr[2];
    pad[3] = p_pr[3];
    conv_param_t<> conv_p = conv_param_t<>(MB, IC, OC, IN_DIM, G, K, stride, pad);
    
    //CALCULATION
    int kernel_dim =
        accumulate(conv_p.K.begin(), conv_p.K.end(), 1, multiplies<int>());
    int KDim = kernel_dim * conv_p.IC;
    int KDimPerGroup = KDim / conv_p.G;
    int OC_per_G = conv_p.OC / conv_p.G;

     // COMPUTING column offset
    std::vector<int32_t>* col_offsets = new std::vector<int32_t>(conv_p.OC);

    for (int g = 0; g < conv_p.G; ++g) {
        col_offsets_with_zero_pt_s8acc32(
            KDimPerGroup,
            OC_per_G,
            OC_per_G,
            reinterpret_cast<std::int8_t*>(B->data) + g * KDimPerGroup * OC_per_G,
            Bint8_zero_point.data(),
            col_offsets->data() + g * OC_per_G,
            conv_p.OC);
    }       
            
    *ret = col_offsets;
}); 


TVM_REGISTER_GLOBAL("tvm.contrib.fbgemm.create_pointer_vector_int")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
    DLTensor* A = args[0];
    int size = args[1];
    std::vector<int32_t>* vec = new std::vector<int32_t>;
    vec->reserve(size);
    int* X = reinterpret_cast<int*>(A->data);
    for (int i = 0; i < size; ++i) {
      vec->data()[i] = X[i];
    }
    *ret = vec;
});

TVM_REGISTER_GLOBAL("tvm.contrib.fbgemm.create_pointer_vector_float")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
    DLTensor* A = args[0];
    int size = args[1];
    std::vector<float>* vec = new std::vector<float>;
    vec->reserve(size);
    float* X = reinterpret_cast<float*>(A->data);
    for (int i = 0; i < size; ++i) {
      vec->data()[i] = X[i];
    }
    *ret = vec;

});


TVM_REGISTER_GLOBAL("tvm.contrib.fbgemm.gemmint8acc32packedwt")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      DLTensor* X = args[0];  // M*K quantized int8 input

      std::uint64_t wt = args[1];
      void* weight = reinterpret_cast<void*>(static_cast<uint64_t>(wt));
      packbmatrix* packB =
          reinterpret_cast<PackBMatrix<std::int8_t, std::int32_t>*>(weight);

      DLTensor* B = args[2];  // N quantized int8 bias
      DLTensor* Y = args[3];
      int threads = args[9];

      //CHECK_EQ(X->ndim, 2);
      // TODO: Ensure correctness here
      // CHECK_EQ(W->ndim, 2);
      // CHECK_EQ(X->shape[1], W->shape[1]);
      //CHECK_EQ(B->ndim, 1);
      // TODO: Ensure correctness here
      // CHECK_EQ(W->shape[0], B->shape[0]);

      float ReQuant_multiplier = (double)args[7];
      std::int32_t x_zero_point = args[4];
      std::int32_t w_zero_point = args[5];
      std::int32_t y_zero_point = args[6];

      int m = X->shape[0];
      int n = Y->shape[1];
      int k = X->shape[1];

      //std::uint64_t co = args[8];
      //void* col_offsts = reinterpret_cast<void*>(static_cast<uint64_t>(co));
      //std::vector<std::int32_t>* column_offsets_ =
      //   reinterpret_cast<std::vector<std::int32_t>*>(col_offsts);

     BlockingFactors params;
     if(args.size() > 10){
        int cntr = 10;
        params.MCB = args[cntr];
        params.NCB = args[cntr + 1];
        params.KCB = args[cntr + 2];
        params.MR = args[cntr + 3];
        params.NR = args[cntr + 4];
        params.NR_MIN = args[cntr + 5];
        params.ROW_INTERLEAVE = args[cntr + 6];


      PackAMatrix<std::uint8_t> packA(
          matrix_op_t::NoTranspose, m, k,
          reinterpret_cast<const std::uint8_t*>(X->data), k, nullptr, 1, &params);

      DoNothing<std::int32_t, std::int32_t> doNothing32BitObj;
      memCopy<> memcopyObj (doNothing32BitObj);

      fbgemmPacked(packA, *packB, reinterpret_cast<std::int32_t*>(Y->data),
                  reinterpret_cast<std::int32_t*>(Y->data), n, memcopyObj, 0,
                   threads, &params); 

	} /*else{

      PackAWithRowOffset<std::uint8_t> packA(
          matrix_op_t::NoTranspose, m, k,
          reinterpret_cast<const std::uint8_t*>(X->data), k, nullptr, 1);

      DoNothing<std::int32_t, std::int32_t> doNothing32BitObj;
      memCopy<> memcopyObj (doNothing32BitObj);

      fbgemmPacked(packA, *packB, reinterpret_cast<std::int32_t*>(Y->data),
                  reinterpret_cast<std::int32_t*>(Y->data), n, memcopyObj, 0,
                   threads); 
     }*/ 
});

bool isValid(BlockingFactors *param)
{
   if (param->MCB % param->MR)
     return false;
   if (param->NCB % param->NR)
     return false;
   if (fbgemmHasAvx512Support()) {
     if (param->MR * (param->NCB / param->NR) > 24)
       return false;
   } else if (fbgemmHasAvx2Support()) {
     if (param->MR * (param->NCB / param->NR) > 16)
       return false;
   }

   return true;

}


TVM_REGISTER_GLOBAL("tvm.contrib.fbgemm.gemmint8acc32packedwt_for_tuning")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      DLTensor* X = args[0];  // M*K quantized int8 input
      DLTensor* W = args[1];  // K*N quantized uint8 weight

      DLTensor* B = args[2];  // N quantized int8 bias
      DLTensor* Y = args[3];
      int threads = args[9];

      float ReQuant_multiplier = (double)args[7];
      std::int32_t x_zero_point = args[4];
      std::int32_t w_zero_point = args[5];
      std::int32_t y_zero_point = args[6];

      int m = X->shape[0];
      int n = Y->shape[1];
      int k = X->shape[1];

      // For tuning these will be garbage values
      std::uint64_t co = args[8];
      //void* col_offsts = reinterpret_cast<void*>(static_cast<uint64_t>(co));
      //std::vector<std::int32_t>* column_offsets_ =
      //    reinterpret_cast<std::vector<std::int32_t>*>(col_offsts);
    
     BlockingFactors params;
     if(args.size() > 10){
        int cntr = 10;
        params.MCB = args[cntr];
        params.NCB = args[cntr + 1];
        params.KCB = args[cntr + 2];
        params.MR = args[cntr + 3];
        params.NR = args[cntr + 4];
        params.NR_MIN = args[cntr + 5];
        params.ROW_INTERLEAVE = args[cntr + 6];

        assert (isValid(&params) == true  && "incorrect configuration");
        

        static PackBMatrix<std::int8_t, std::int32_t> packB_ (
            matrix_op_t::NoTranspose, k, n,
            reinterpret_cast<const std::int8_t*>(W->data), n, nullptr, 1,
            &params);

      PackAMatrix<std::uint8_t> packA(
          matrix_op_t::NoTranspose, m, k,
          reinterpret_cast<const std::uint8_t*>(X->data), k, nullptr, 1, 
          &params);

      DoNothing<std::int32_t, std::int32_t> doNothing32BitObj;
      memCopy<> memcopyObj (doNothing32BitObj);

      fbgemmPacked(packA, packB_, reinterpret_cast<std::int32_t*>(Y->data),
                  reinterpret_cast<std::int32_t*>(Y->data), n, memcopyObj, 0,
                   threads, &params); 
	}
    });


/*
 Supports prepacked weight matrix B and requantization. 
 It will receive a pointer for prepacked weight directly as its argument;
 It will also receive a pointer for col_offsets and other parameters for
 requantization.
*/
TVM_REGISTER_GLOBAL("tvm.contrib.fbgemm.gemmint8acc32packedwt_with_requant")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      DLTensor* X = args[0];  // M*K quantized int8 input
      std::uint64_t wt = args[1];
      void* weight = reinterpret_cast<void*>(static_cast<uint64_t>(wt));
      packbmatrix* packB =
          reinterpret_cast<PackBMatrix<std::int8_t, std::int32_t>*>(weight);

      DLTensor* B = args[2];  // N quantized int8 bias
      DLTensor* Y = args[3];
      int threads = args[9];

      //CHECK_EQ(X->ndim, 2);
      //CHECK_EQ(W->ndim, 2);
      //CHECK_EQ(B->ndim, 1);
      //CHECK_EQ(X->shape[1], W->shape[1]);
      //CHECK_EQ(W->shape[0], B->shape[0]);

      float ReQuant_multiplier = (double)args[7];
      std::int32_t x_zero_point = args[4];
      std::int32_t w_zero_point = args[5];
      std::int32_t y_zero_point = args[6];

      int m = X->shape[0];
      int n = Y->shape[1];
      int k = X->shape[1];

      BlockingFactors params;

      if(args.size() > 10) {
        int cntr = 10;
        params.MCB = args[cntr];
        params.NCB = args[cntr + 1];
        params.KCB = args[cntr + 2];
        params.MR = args[cntr + 3];
        params.NR = args[cntr + 4];
        params.NR_MIN = args[cntr + 5];
        params.ROW_INTERLEAVE = args[cntr + 6];
      }

      std::vector<std::int32_t> row_offsets_(
          PackAWithRowOffset<uint8_t>::rowOffsetBufferSize());
      std::vector<std::int32_t> Y_int32_(n * m);

      std::uint64_t co_addr = args[8];
      void* co = reinterpret_cast<void*>(static_cast<uint64_t>(co_addr));
      std::vector<std::int32_t>* column_offsets_ =
          reinterpret_cast<std::vector<std::int32_t>*>(co);

      std::vector<TensorQuantizationParams> temp_qparams;
      temp_qparams.push_back(TensorQuantizationParams{1.0, w_zero_point});

      if(args.size() > 10){

        PackAWithRowOffset<std::uint8_t> packA(
            matrix_op_t::NoTranspose, m, k,
            reinterpret_cast<const std::uint8_t*>(X->data), k, nullptr, 1,
            row_offsets_.data(), &params);

        DoNothing<> doNothingObj{};
        ReQuantizeOutput<false> outputProcObj(
            doNothingObj, &ReQuant_multiplier, y_zero_point, x_zero_point,
            &w_zero_point, packA.getRowOffsetBuffer(), (*column_offsets_).data(),
            reinterpret_cast<const std::int32_t*>(B->data), n);

        fbgemmPacked(packA, *packB, reinterpret_cast<std::uint8_t*>(Y->data),
                     Y_int32_.data(), n, outputProcObj, 0, threads,
                     &params);  // num_threads

      }  else {

        PackAWithRowOffset<std::uint8_t> packA(
            matrix_op_t::NoTranspose, m, k,
            reinterpret_cast<const std::uint8_t*>(X->data), k, nullptr, 1,
            row_offsets_.data());

        DoNothing<> doNothingObj{};
        ReQuantizeOutput<false> outputProcObj(
            doNothingObj, &ReQuant_multiplier, y_zero_point, x_zero_point,
            &w_zero_point, packA.getRowOffsetBuffer(), (*column_offsets_).data(),
            reinterpret_cast<const std::int32_t*>(B->data), n);

        fbgemmPacked(packA, *packB, reinterpret_cast<std::uint8_t*>(Y->data),
                     Y_int32_.data(), n, outputProcObj, 0,
                     threads);  // num_threads
      }
});

TVM_REGISTER_GLOBAL("tvm.contrib.fbgemm.fully_connected_int8")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      DLTensor* X = args[0];  // M*K quantized int8 input
      DLTensor* W = args[1];  // N*K quantized int8 weight
      DLTensor* B = args[2];  // N quantized int8 bias
      // ignore the axis and axis_w now for testing purpose
      DLTensor* Y = args[3];
      int threads = args[8];

      CHECK_EQ(X->ndim, 2);
      CHECK_EQ(W->ndim, 2);
      CHECK_EQ(B->ndim, 1);
      CHECK_EQ(X->shape[1], W->shape[1]);
      CHECK_EQ(W->shape[0], B->shape[0]);

      float ReQuant_multiplier = (double)args[7];
      std::int32_t x_zero_point = args[4];
      std::int32_t w_zero_point = args[5];
      std::int32_t y_zero_point = args[6];

      int m = X->shape[0];
      int n = W->shape[0];
      int k = X->shape[1];

      BlockingFactors params;

      if (args.size() > 9) {
        params.MCB = args[9];
        params.NCB = args[10];
        params.KCB = args[11];
        params.MR = args[12];
        params.NR = args[13];
        params.NR_MIN = args[14];
        params.ROW_INTERLEAVE = args[15];
      }

      std::vector<std::int32_t> row_offsets_(
          PackAWithRowOffset<uint8_t>::rowOffsetBufferSize());
      std::vector<std::int32_t> Y_int32_(n * m);
      std::vector<std::int32_t> column_offsets_;

      std::vector<TensorQuantizationParams> temp_qparams;
      temp_qparams.push_back(TensorQuantizationParams{1.0, w_zero_point});

      if (args.size() > 9) {
        PackBMatrix<std::int8_t, std::int32_t> packB(
            matrix_op_t::Transpose, k, n,
            reinterpret_cast<const std::int8_t*>(W->data), k, nullptr, 1,
            &params);

        PackAWithRowOffset<std::uint8_t> packA(
            matrix_op_t::NoTranspose, m, k,
            reinterpret_cast<const std::uint8_t*>(X->data), k, nullptr, 1,
            row_offsets_.data(), &params);

        ComputeColumnOffsets<std::int8_t>(
            k, n, reinterpret_cast<const std::int8_t*>(W->data), temp_qparams,
            column_offsets_);

        DoNothing<> doNothingObj{};
        ReQuantizeOutput<false> outputProcObj(
            doNothingObj, &ReQuant_multiplier, y_zero_point, x_zero_point,
            &w_zero_point, packA.getRowOffsetBuffer(), column_offsets_.data(),
            reinterpret_cast<const std::int32_t*>(B->data), n);

        fbgemmPacked(packA, packB, reinterpret_cast<std::uint8_t*>(Y->data),
                     Y_int32_.data(), n, outputProcObj, 0, threads,
                     &params);  // num_threads

      } else {
        PackBMatrix<std::int8_t, std::int32_t> packB(
            matrix_op_t::Transpose, k, n,
            reinterpret_cast<const std::int8_t*>(W->data), k, nullptr, 1);

        PackAWithRowOffset<std::uint8_t> packA(
            matrix_op_t::NoTranspose, m, k,
            reinterpret_cast<const std::uint8_t*>(X->data), k, nullptr, 1,
            row_offsets_.data());

        ComputeColumnOffsets<std::int8_t>(
            k, n, reinterpret_cast<const std::int8_t*>(W->data), temp_qparams,
            column_offsets_);

        DoNothing<> doNothingObj{};
        ReQuantizeOutput<false> outputProcObj(
            doNothingObj, &ReQuant_multiplier, y_zero_point, x_zero_point,
            &w_zero_point, packA.getRowOffsetBuffer(), column_offsets_.data(),
            reinterpret_cast<const std::int32_t*>(B->data), n);

        fbgemmPacked(packA, packB, reinterpret_cast<std::uint8_t*>(Y->data),
                     Y_int32_.data(), n, outputProcObj, 0,
                     threads);  // num_threads
      }
    });


TVM_REGISTER_GLOBAL("tvm.contrib.fbgemm.conv_int8")
    .set_body([](TVMArgs args, TVMRetValue* ret) {

    DLTensor* A = args[0];

    std::uint64_t wt = args[1];
    void* weight = reinterpret_cast<void*>(static_cast<uint64_t>(wt));
    PackWeightsForConv<2>* packedB =
        reinterpret_cast<PackWeightsForConv<2>*>(weight);

    DLTensor* Y = args[2];
    std::int32_t Aint8_zero_point = args[3];
    
    std::int32_t Bint8_zp = args[4];
    aligned_vector<int32_t> Bint8_zero_point = {Bint8_zp};

    std::int32_t C_zero_point = args[5];

    float num = (double) args[6];
    aligned_vector<float> C_multiplier = {num};

    std::uint64_t co_addr = args[7];
    void* co = reinterpret_cast<void*>(static_cast<uint64_t>(co_addr));
    std::vector<std::int32_t>* column_offsets_ =
        reinterpret_cast<std::vector<std::int32_t>*>(co);

    int cntr = 8;
    int MB = args[cntr];
    int IC = args[cntr + 1];
    int OC = args[cntr + 2];

    std::uint64_t id_addr = args[cntr + 3];
    void* id_d = reinterpret_cast<void*>(static_cast<uint64_t>(id_addr));
    std::vector<int>* IN_DIM_v =
        reinterpret_cast<std::vector<int>*>(id_d);
    std::array<int, 2> IN_DIM;
    IN_DIM[0] = IN_DIM_v->data()[0];
    IN_DIM[1] = IN_DIM_v->data()[1];

    int G = args[cntr + 4];

    std::uint64_t k_addr = args[cntr + 5];
    void* k_d = reinterpret_cast<void*>(static_cast<uint64_t>(k_addr));
    std::vector<int>* K_v =
        reinterpret_cast<std::vector<int>*>(k_d);
    std::array<int, 2> K;
    K[0] = K_v->data()[0];
    K[1] = K_v->data()[1];

    std::uint64_t s_addr = args[cntr + 6];
    void* s_d = reinterpret_cast<void*>(static_cast<uint64_t>(s_addr));
    std::vector<int>* stride_v =
        reinterpret_cast<std::vector<int>*>(s_d);
    std::array<int, 2> stride;
    stride[0] = stride_v->data()[0];
    stride[1] = stride_v->data()[1];

    std::uint64_t p_addr = args[cntr + 7];
    void* p_d = reinterpret_cast<void*>(static_cast<uint64_t>(p_addr));
    std::vector<int>* p_v =
        reinterpret_cast<std::vector<int>*>(p_d);
    std::array<int, 4> pad;
    pad[0] = p_v->data()[0];
    pad[1] = p_v->data()[1];
    pad[2] = p_v->data()[2];
    pad[3] = p_v->data()[3];

    conv_param_t<2> conv_p(MB, IC, OC, IN_DIM, G, K, stride, pad);

    CHECK_EQ(conv_p.IC % conv_p.G, 0);
    CHECK_EQ(conv_p.OC % conv_p.G, 0);

    int kernel_dim =
        accumulate(conv_p.K.begin(), conv_p.K.end(), 1, multiplies<int>());

    int im_out_dim = accumulate(
        conv_p.OUT_DIM.begin(), conv_p.OUT_DIM.end(), 1, multiplies<int>());

    int im_dim = accumulate(
        conv_p.IN_DIM.begin(), conv_p.IN_DIM.end(), 1, multiplies<int>());

    int KDim = kernel_dim * conv_p.IC;
    int KDimPerGroup = KDim / conv_p.G;
    int OC_per_G = conv_p.OC / conv_p.G;

    std::vector<std::int32_t>* Y_int32_ =
    new std::vector<int32_t>(conv_p.MB * im_out_dim * conv_p.OC);

    // no-op output process objects
    DoNothing<> doNothingObj{};
    ReQuantizeOutput<false, QuantizationGranularity::TENSOR> outputProcObj(
        doNothingObj,
        C_multiplier.data(),
        C_zero_point,
        Aint8_zero_point,
        Bint8_zero_point.data(),
        nullptr, // row offsets
        column_offsets_->data(),
        nullptr, // bias
        conv_p.OC,
        conv_p.G);

    fbgemmConv(
        conv_p,
        reinterpret_cast<const std::uint8_t*>(A->data),
        *packedB,
        reinterpret_cast<std::uint8_t*>(Y->data),
        Y_int32_->data(),
        outputProcObj,
        0,
        1);

});

}  // namespace contrib
}  // namespace tvm
