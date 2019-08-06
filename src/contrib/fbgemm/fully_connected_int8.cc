/*!
 *  Copyright (c) 2019 by Contributors
 * \file Use external fbgemm library call.
 */

#include <cpuinfo.h>
#include <dmlc/logging.h>
#include <fbgemm/Fbgemm.h>
#include <fbgemm/FbgemmFP16.h>
#include <fbgemm/QuantUtilsAvx2.h>
#include <fbgemm/AlignedVec.h>
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
template <>
struct extension_class_info<packbmatrix> {
  static const int code = 19;
};

TVM_REGISTER_EXT_TYPE(packbmatrix);
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

TVM_REGISTER_GLOBAL("tvm.contrib.fbgemm.packB_with_allocated_tensor")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      //This in the input weight matrix which needs to be packed		    
      DLTensor* inputB = args[0];  
      //This is the iraw memory for packed weights
      DLTensor* packedBufferB = args[1];  
      int num_rows = inputB->shape[0];
      int num_cols = inputB->shape[1];
      PackBMatrix<std::int8_t, std::int32_t> *packB;

      //Assume input is not transposed
      static BlockingFactors params;
      if (args.size() > 2) {
        int cntr = 2;
        params.MCB = args[cntr];
        params.NCB = args[cntr + 1];
        params.KCB = args[cntr + 2];
        params.MR = args[cntr + 3];
        params.NR = args[cntr + 4];
        params.NR_MIN = args[cntr + 5];
        params.ROW_INTERLEAVE = args[cntr + 6];
  
        packB = new PackBMatrix<std::int8_t, std::int32_t>(
            matrix_op_t::NoTranspose, num_rows, num_cols,
            reinterpret_cast<const std::int8_t*>(inputB->data), num_cols, 
	    reinterpret_cast<std::int8_t*>(packedBufferB->data), 1,
            &params);
        } else {
        packB = new PackBMatrix<std::int8_t, std::int32_t>(
            matrix_op_t::NoTranspose, num_rows, num_cols,
            reinterpret_cast<const std::int8_t*>(inputB->data), num_cols, 
	    reinterpret_cast<std::int8_t*>(packedBufferB->data), 1);
      }

    });
/*
TVM_REGISTER_GLOBAL("tvm.contrib.fbgemm.create_packed_matrix_with_buffer")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      //This in the input weight matrix which needs to be packed		    
      DLTensor* inputB = args[0];  
      int k = inputB->shape[0];
      int n = inputB->shape[1];

      auto packB = new PackBMatrix<std::int8_t, std::int32_t>(
          matrix_op_t::NoTranspose, 
	  256,
	  32,
	  256,
	  32,
	  1,
	  1,
	  32,
          reinterpret_cast<std::int8_t*>(inputB->data)); 
 
    });

//This function does a gemm where the weights have been pre-packed
//This implementation does not depend on opaque pointers
TVM_REGISTER_GLOBAL("tvm.contrib.fbgemm.packedgemm_U8S8ACC32")
    .set_body([](TVMArgs args, TVMRetValue* ret) {

      DLTensor* A = args[0];  // M*K quantized uint8 input
      DLTensor* PackedB = args[1];  // K*N quantized int8 input
      DLTensor* C = args[2];  // M*N int32 output

      int m = A->shape[0];
      int n = PackedB->shape[1];
      int k = A->shape[1];

      auto packB = PackBMatrix<std::int8_t, std::int32_t>(
                matrix_op_t::NoTranspose, 
	        256,
	        32,
	        256,
	        32,
	        1,
	        1,
	        32,
                reinterpret_cast<std::int8_t*>(PackedB->data)); 

      PackAMatrix<std::uint8_t> packA(
          matrix_op_t::NoTranspose, m, k,
          reinterpret_cast<const std::uint8_t*>(A->data), k, nullptr, 1);

      DoNothing<std::int32_t, std::int32_t> doNothing32BitObj;
      memCopy<> memcopyObj (doNothing32BitObj);

      fbgemmPacked(packA, packB, reinterpret_cast<std::int32_t*>(C->data),
                  reinterpret_cast<std::int32_t*>(C->data), n, memcopyObj, 0, 1); 

 });	
*/
TVM_REGISTER_GLOBAL("tvm.contrib.fbgemm.pack_matrixB_int8")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      DLTensor* W = args[0];  // N*K quantized int8 weight
      int threads = args[1];

      CHECK_EQ(W->ndim, 2);

      int k = W->shape[0];
      int n = W->shape[1];

      BlockingFactors params;
      if (args.size() > 2) {
        int cntr = 2;
        params.MCB = args[cntr];
        params.NCB = args[cntr + 1];
        params.KCB = args[cntr + 2];
        params.MR = args[cntr + 3];
        params.NR = args[cntr + 4];
        params.NR_MIN = args[cntr + 5];
        params.ROW_INTERLEAVE = args[cntr + 6];
  
        auto packB = new PackBMatrix<std::int8_t, std::int32_t>(
            matrix_op_t::NoTranspose, k, n,
            reinterpret_cast<const std::int8_t*>(W->data), n, nullptr, 1,
            &params);
       //packB->printPackedMatrix("packingB");
        *ret = packB;
      } else {
        auto packB = new PackBMatrix<std::int8_t, std::int32_t>(
            matrix_op_t::NoTranspose, k, n,
            reinterpret_cast<const std::int8_t*>(W->data), n, nullptr, 1);
        *ret = packB;
      }
    });

TVM_REGISTER_GLOBAL("tvm.contrib.fbgemm.compute_col_offsets_int8")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      DLTensor* W = args[0];  // N*K quantized int8 weight
      int threads = args[1];
      std::int32_t w_zero_point = args[2];

      CHECK_EQ(W->ndim, 2);
      int n = W->shape[0];
      int k = W->shape[1];

      std::vector<TensorQuantizationParams> temp_qparams;
      temp_qparams.push_back(TensorQuantizationParams{1.0, w_zero_point});

      std::vector<std::int32_t>* column_offsets_ =
          new std::vector<std::int32_t>;
      ComputeColumnOffsets<std::int8_t>(
          k, n, reinterpret_cast<const std::int8_t*>(W->data), temp_qparams,
          *column_offsets_);
      *ret = column_offsets_;
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


void transposeConvWeights(
    const conv_param_t<2>& conv_p,
    const std::int8_t* src,
    std::int8_t* dest) {
  int G = conv_p.G;
  int IC_per_G = conv_p.IC / conv_p.G;
  int OC_per_G = conv_p.OC / conv_p.G;

  int R = conv_p.K[0];
  int S = conv_p.K[1];
  // Transforms weights from  G K/G (R S C/G) to G (R S C/G) K/G format.
  for (int r = 0; r < R; ++r) {
    for (int s = 0; s < S; ++s) {
      for (int k = 0; k < OC_per_G; ++k) {
        for (int g = 0; g < G; ++g) {
          for (int c = 0; c < IC_per_G; ++c) {
            dest[(((g * R + r) * S + s) * IC_per_G + c) * OC_per_G + k] =
                src[(((g * OC_per_G + k) * R + r) * S + s) * IC_per_G + c];
            }
          }
        }
      }
    }
}


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
    int cntr = 2;
    int MB = args[cntr];
    int IC = args[cntr + 1];
    int OC = args[cntr + 2];

    std::array<int, 2> IN_DIM = {0, 0};
    IN_DIM[0] = args[cntr + 3];
    IN_DIM[1] = args[cntr + 4];

    int G = args[cntr + 5];

    std::array<int, 2> K = {0, 0};
    K[0] = args[cntr + 6];
    K[1] = args[cntr + 7];

    std::array<int, 2> stride = {0, 0};
    stride[0] = args[cntr + 8];
    stride[1] = args[cntr + 9];

    std::array<int, 4> pad = {0, 0, 0, 0};
    pad[0] = args[cntr + 10];
    pad[1] = args[cntr + 11];
    pad[2] = args[cntr + 12];
    pad[3] = args[cntr + 13];
    conv_param_t<> conv_p = conv_param_t<>(MB, IC, OC, IN_DIM, G, K, stride, pad);

    //CALCULATION
    int kernel_dim = conv_p.K[0] * conv_p.K[1];
        //accumulate(conv_p.K.begin(), conv_p.K.end(), 1, multiplies<int>());
    int KDim = kernel_dim * conv_p.IC;
    int KDimPerGroup = KDim / conv_p.G;
    int OC_per_G = conv_p.OC / conv_p.G;

    std::vector<int8_t> Bint8_tr(kernel_dim * conv_p.IC * (conv_p.OC / conv_p.G));

    transposeConvWeights(conv_p, reinterpret_cast<std::int8_t*>(B->data), Bint8_tr.data());

     // COMPUTING column offset
    std::vector<int32_t>* col_offsets = new std::vector<int32_t>(conv_p.OC);

    for (int g = 0; g < conv_p.G; ++g) {
        col_offsets_with_zero_pt_s8acc32(
            KDimPerGroup,
            OC_per_G,
            OC_per_G,
            Bint8_tr.data() + g * KDimPerGroup * OC_per_G,
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

TVM_REGISTER_GLOBAL("tvm.contrib.fbgemm.pack_matrixB_int8_conv")
    .set_body([](TVMArgs args, TVMRetValue* ret) {

    DLTensor* W = args[0];

    int cntr = 1;
    int MB = args[cntr];
    int IC = args[cntr + 1];
    int OC = args[cntr + 2];

    std::array<int, 2> IN_DIM = {0, 0};
    IN_DIM[0] = args[cntr + 3];
    IN_DIM[1] = args[cntr + 4];

    int G = args[cntr + 5];

    std::array<int, 2> K = {0, 0};
    K[0] = args[cntr + 6];
    K[1] = args[cntr + 7];

    std::array<int, 2> stride = {0, 0};
    stride[0] = args[cntr + 8];
    stride[1] = args[cntr + 9];

    std::array<int, 4> pad = {0, 0, 0, 0};
    pad[0] = args[cntr + 10];
    pad[1] = args[cntr + 11];
    pad[2] = args[cntr + 12];
    pad[3] = args[cntr + 13];

    conv_param_t<> conv_p = conv_param_t<>(MB, IC, OC, IN_DIM, G, K, stride, pad);

    BlockingFactors params;

    if (args.size() > 16) {
      int cntr = 15;
      params.MCB = args[cntr];
      params.NCB = args[cntr + 1];
      params.KCB = args[cntr + 2];
      params.MR = args[cntr + 3];
      params.NR = args[cntr + 4];
      params.NR_MIN = args[cntr + 5];
      params.ROW_INTERLEAVE = args[cntr + 6];
      PackWeightsForConv<2>* packedB =
      new PackWeightsForConv<2>(conv_p, reinterpret_cast<std::int8_t*>(W->data), &params);
      *ret = packedB;

    } else {
      PackWeightsForConv<2>* packedB =
      new PackWeightsForConv<2>(conv_p, reinterpret_cast<std::int8_t*>(W->data));
      *ret = packedB;
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
    DLTensor* B = args[8];
    int cntr = 9;
    int MB = args[cntr];
    int IC = args[cntr + 1];
    int OC = args[cntr + 2];

    std::array<int, 2> IN_DIM = {0, 0};
    IN_DIM[0] = args[cntr + 3];
    IN_DIM[1] = args[cntr + 4];

    int G = args[cntr + 5];

    std::array<int, 2> K = {0, 0};
    K[0] = args[cntr + 6];
    K[1] = args[cntr + 7];

    std::array<int, 2> stride = {0, 0};
    stride[0] = args[cntr + 8];
    stride[1] = args[cntr + 9];

    std::array<int, 4> pad = {0, 0, 0, 0};
    pad[0] = args[cntr + 10];
    pad[1] = args[cntr + 11];
    pad[2] = args[cntr + 12];
    pad[3] = args[cntr + 13];

    conv_param_t<2> conv_p(MB, IC, OC, IN_DIM, G, K, stride, pad);

    CHECK_EQ(conv_p.IC % conv_p.G, 0);
    CHECK_EQ(conv_p.OC % conv_p.G, 0);

    BlockingFactors params;
    if(args.size() > cntr + 16) {
        params.MCB = args[cntr + 15];
        params.NCB = args[cntr + 16];
        params.KCB = args[cntr + 17];
        params.MR = args[cntr + 18];
        params.NR = args[cntr + 19];
        params.NR_MIN = args[cntr + 20];
        params.ROW_INTERLEAVE = args[cntr + 21];
    }
    int kernel_dim = conv_p.K[0] * conv_p.K[1];
        //accumulate(conv_p.K.begin(), conv_p.K.end(), 1, multiplies<int>());

    int im_out_dim = conv_p.OUT_DIM[0] * conv_p.OUT_DIM[1];//accumulate(
        //conv_p.OUT_DIM.begin(), conv_p.OUT_DIM.end(), 1, multiplies<int>());

    int im_dim = conv_p.IN_DIM[0] * conv_p.IN_DIM[1];//accumulate(
        //conv_p.IN_DIM.begin(), conv_p.IN_DIM.end(), 1, multiplies<int>());

    int KDim = kernel_dim * conv_p.IC;
    int MDim = conv_p.MB * im_out_dim;
    int NDim = conv_p.OC / conv_p.G;
    int KDimPerGroup = KDim / conv_p.G;
    int OC_per_G = conv_p.OC / conv_p.G;
    static std::vector<std::int32_t> Y_int32_(conv_p.MB * im_out_dim * conv_p.OC);
    //std::fill(Y_int32_.begin(), Y_int32_.end(), 0);

if (args.size() > cntr + 16) {
        
    static int count = 1;
    static std::vector<int32_t> col_offsets(conv_p.OC);
    if (count == 1) {
    count += 1;
    for (int g = 0; g < conv_p.G; ++g) {
        col_offsets_with_zero_pt_s8acc32(
            KDimPerGroup,
            OC_per_G,
            OC_per_G,
            reinterpret_cast<std::int8_t*>(B->data) + g * KDimPerGroup * OC_per_G,
            Bint8_zero_point.data(),
            col_offsets.data() + g * OC_per_G,
            conv_p.OC);
    }            
    }     
    static PackWeightsForConv<2> packedBmat(conv_p, reinterpret_cast<std::int8_t*>(B->data), &params);
    // no-op output process objects
    DoNothing<> doNothingObj{};
    ReQuantizeOutput<false, QuantizationGranularity::TENSOR> outputProcObj(
        doNothingObj,
        C_multiplier.data(),
        C_zero_point,
        Aint8_zero_point,
        Bint8_zero_point.data(),
        nullptr, // row offsets
        col_offsets.data(),//col_offsets.data(),
        nullptr, // bias
        conv_p.OC,
        conv_p.G);
    fbgemmConv(
        conv_p,
        reinterpret_cast<const std::uint8_t*>(A->data),
        packedBmat,//*packedB,//packedBmat,
        reinterpret_cast<std::uint8_t*>(Y->data),
        Y_int32_.data(),
        outputProcObj,
        0,
        1, &params);
} else {
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
//std::chrono::time_point<std::chrono::high_resolution_clock> begin, end;
//    double ttot = 0.0;
//      begin = std::chrono::high_resolution_clock::now();
    fbgemmConv(
        conv_p,
        reinterpret_cast<const std::uint8_t*>(A->data),
        *packedB,
        reinterpret_cast<std::uint8_t*>(Y->data),
        Y_int32_.data(),
        outputProcObj,
        0,
        1);
//end = std::chrono::high_resolution_clock::now();
//        auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
//        ttot += dur.count();
//std::cout << 2 * MDim * NDim * KDim / ttot << std::endl;
}
});


}  // namespace contrib
}  // namespace tvm
