import logging
import argparse
import os
import mxnet as mx
from mxnet import gluon
from mxnet.gluon.model_zoo import vision
import numpy as np
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
import tvm.contrib.graph_runtime as runtime
from tvm import autotvm, relay


# def evaluate(args, graph, lib, params, ctx):
#     """Evaluate on the validation set."""
#     import tvm
#     from tvm.contrib import graph_runtime
# 
#     # tetup dataset.
#     batch_size = args.batch_size
# 
#     # create runtime module
#     # m = graph_runtime.create(graph, lib, ctx)
#     m = debug_runtime.create(graph, lib, ctx)
#     # logging.debug(graph.symbol().debug_str())
#     oshape = (batch_size, args.num_classes)
#     ishape = (args.batch_size, 3, 224, 224)
#     m.set_input('data', tvm.nd.array(np.random.uniform(size=(ishape)).astype("float32")))
#     m.set_input(**params)
#     m.run()
# 
#     out = m.get_output(0).asnumpy()
#     # print(out)
#     ftimer = m.module.time_evaluator("run", ctx, 10)
#     prof_res = ftimer()
#     print("TVM time: ", prof_res.mean)

def get_network(args, gluon_model):
    """Get the symbol definition and random weight of a network"""
    input_shape = (1, 3, 224, 224)
    output_shape = (1, 1000)
    gluon_model = vision.get_model(args.model, pretrained=True)
    net, params = relay.frontend.from_mxnet(gluon_model, {"data": input_shape})
    
    return net, params, input_shape, output_shape

num_threads = 1
os.environ["TVM_NUM_THREADS"] = str(num_threads)

tuning_option = {
    'log_filename':"resnet50_v1",
    'tuner': 'random',
    'early_stopping': None,

    'measure_option': autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(number=10, repeat=1,
                                   min_repeat_ms=1000),
    ),
}

# You can skip the implementation of this function for this tutorial.
def tune_kernels(tasks,
                 args,
                 measure_option,
                 tuner='gridsearch',
                 early_stopping=None,
                 log_filename='tuning.log'):

    for i, tsk in enumerate(tasks):
        prefix = "[Task %2d/%2d] " % (i+1, len(tasks))

        # converting conv2d tasks to conv2d_NCHWc tasks
        op_name = tsk.workload[0]
        if op_name == 'conv2d':
            func_create = 'topi_x86_conv2d_NCHWc'
        else:
            raise ValueError("Tuning {} is not supported on x86".format(op_name))

        task = autotvm.task.create(func_create, args=tsk.args,
                                   target=args.target, template_key='direct')
        task.workload = tsk.workload

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(task, loss_type='rank')
        elif tuner == 'ga':
            tuner_obj = GATuner(task, pop_size=50)
        elif tuner == 'random':
            tuner_obj = RandomTuner(task)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(task)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        # do tuning
        n_trial=len(task.config_space)
        tuner_obj.tune(n_trial=n_trial,
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(n_trial, prefix=prefix),
                           autotvm.callback.log_to_file(log_filename)])


def tune_and_evaluate(args, gluon_model, tuning_opt):
    # extract workloads from relay program
    print("Extract tasks...")
   
    net, params, data_shape, out_shape = get_network(args, gluon_model)

    tasks = autotvm.task.extract_from_program(net, target=args.target,
                                              params=params, ops=(relay.op.nn.conv2d,))

    # run tuning tasks
    print("Tuning...")
    tune_kernels(tasks, args, **tuning_opt)

    # compile kernels with history best records
    with autotvm.apply_history_best(log_file):
        print("Compile...")
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build_module.build(
                net, target=target,  params=params)

        # upload parameters to device
        ctx = tvm.cpu()
        data_tvm = tvm.nd.array((np.random.uniform(size=data_shape)).astype(dtype))
        module = runtime.create(graph, lib, ctx)
        module.set_input('data', data_tvm)
        module.set_input(**params)

        # evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", ctx, number=100, repeat=3)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
              (np.mean(prof_res), np.std(prof_res)))


### def build_model(args, gluon_model):
###     """Build with relay."""
###     import tvm
###     from tvm import relay
###     from tvm.relay import quantize as qtz
###     img_size = 299 if args.model == 'inceptionv3' else 224
###     data_shape = (args.batch_size, 3, img_size, img_size)
###     net, params = relay.frontend.from_mxnet(gluon_model, {"data": data_shape})
###     target = args.target
### 
###     if args.original:
###         # run original model
###         with relay.build_config(opt_level=3):
###             graph, lib, params = relay.build(net, target, params=params)
###         ctx = tvm.nd.context(target, 0)
###         return graph, lib, params, ctx
### 
###     # constant folding and scale folding.
###     print('original')
###     print(net.astext(show_meta_data=False))
###     with relay.build_config(opt_level=2):
###         qgraph = relay.optimize(net, target, params)
###         # qgraph = relay.optimize(qgraph)
###     print('after optimize')
###     print(qgraph.astext(show_meta_data=False))
### 
###     with qtz.qconfig(skip_k_conv=0,
###                      nbit_input=args.nbit_input,
###                      nbit_weight=args.nbit_input,
###                      global_scale=args.global_scale,
###                      dtype_input=args.dtype_input,
###                      dtype_weight=args.dtype_weight,
###                      dtype_activation=args.dtype_output,
###                      store_lowbit_output=False,
###                      debug_enabled_ops=None):
###         print(qtz.current_qconfig())
###         qgraph = qtz.annotate(qgraph)
###         print('after annotate')
###         print(qgraph.astext(show_meta_data=False))
###         qgraph = qtz.calibrate(qgraph)
###         print('after calibrate\n')
###         print(qgraph.astext(show_meta_data=False))
###         if not args.simulated:
###             qgraph = qtz.realize(qgraph)
###             qgraph = relay.ir_pass.infer_type(qgraph)
###             print('after realize\n')
###             print(qgraph.astext(show_meta_data=False))
### 
###     with relay.build_config(opt_level=3):
###         graph, lib, params = relay.build(qgraph, target)
###         tasks = autotvm.task.extract_from_graph(
###             ggraph,
###             target=args.target,
###                
###         )
###     ctx = tvm.nd.context(target, 0)
###     return graph, lib, params, ctx


def main(args):
    gluon_model = vision.get_model(args.model, pretrained=True)
    tune_and_evaluate(args, gluon_model, tuning_option)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ImageNet validation accuracy")
    parser.add_argument("--rec-val", type=str, default="~/.mxnet/datasets/imagenet/rec/val.rec",
                        help="the validation data")
    parser.add_argument("--num-classes", type=int, default=1000,
                        help="batch size")
    parser.add_argument("--model", type=str, default="resnet50_v1",
                        help="Name of the model")
    parser.add_argument("--log-interval", type=int, default=100,
                        help="log interval")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="batch size")
    parser.add_argument("--target", type=str, default="llvm -mcpu=skylake-avx512",
                        help="target option")
    parser.add_argument("--nbit-input", type=int, default=8,
                        help="number of input bits")
    parser.add_argument("--nbit-output", type=int, default=32,
                        help="number of output bits")
    parser.add_argument("--dtype-input", type=str, default="uint8",
                        help="number of input bits")
    parser.add_argument("--dtype-weight", type=str, default="int8",
                        help="number of input bits for weight")
    parser.add_argument("--dtype-output", type=str, default="int32",
                        help="number of output bits")
    parser.add_argument("--global-scale", type=float, default=8.0,
                        help="global activation scale")
    parser.add_argument("--original", action="store_true",
                        help='whether to use original graph')
    parser.add_argument("--simulated", action="store_true",
                        help='whether to use simulated graph')
    args = parser.parse_args()
    # logging.basicConfig(level=logging.DEBUG)
    logging.info(args)
    main(args)
