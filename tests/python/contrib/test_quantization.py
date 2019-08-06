import logging
import argparse
import os
import mxnet as mx
from mxnet import gluon
from mxnet.gluon.model_zoo import vision
#from gluoncv2.model_provider import get_model as glcv2_get_model
import numpy as np
from tvm.contrib.debugger import debug_runtime


num_threads = 1
os.environ["TVM_NUM_THREADS"] = str(num_threads)

def evaluate(args, graph, lib, params, ctx):
    """Evaluate on the validation set."""
    import tvm
    from tvm.contrib import graph_runtime

    # tetup dataset.
    batch_size = args.batch_size

    # create runtime module
    # m = graph_runtime.create(graph, lib, ctx)
    m = debug_runtime.create(graph, lib, ctx, dump_root="/tmp/tvmdbg")
    print("TVM model running...")
    # logging.debug(graph.symbol().debug_str())
    oshape = (batch_size, args.num_classes)
    ishape = (args.batch_size, 3, 224, 224)
    m.set_input('data', tvm.nd.array(np.random.uniform(size=(ishape)).astype("float32")))
    m.set_input(**params)
    m.run()

    out = m.get_output(0).asnumpy()
    # print(out)
    ftimer = m.module.time_evaluator("run", ctx, 1)
    prof_res = ftimer()
    print("TVM time: ", prof_res.mean)


def build_model(args, gluon_model):
    """Build with relay."""
    import tvm
    from tvm import relay
    from tvm.relay import quantize as qtz
    img_size = 299 if args.model == 'inceptionv3' else 224
    data_shape = (args.batch_size, 3, img_size, img_size)
    net, params = relay.frontend.from_mxnet(gluon_model, {"data": data_shape})
    target = args.target

    if args.original:
        # run original model
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build(net, target, params=params)
        ctx = tvm.nd.context(target, 0)
        return graph, lib, params, ctx

    # constant folding and scale folding.
    print('original')
    print(net.astext(show_meta_data=False))
    with relay.build_config(opt_level=2):
        qgraph = relay.optimize(net, target, params)
        # qgraph = relay.optimize(qgraph)
    print('after optimize')
    print(qgraph.astext(show_meta_data=False))

    with qtz.qconfig(skip_k_conv=1,
                     nbit_input=args.nbit_input,
                     nbit_weight=args.nbit_input,
                     global_scale=args.global_scale,
                     dtype_input=args.dtype_input,
                     dtype_weight=args.dtype_weight,
                     dtype_activation=args.dtype_output,
                     store_lowbit_output=False,
                     debug_enabled_ops=None):
        print(qtz.current_qconfig())
        qgraph = qtz.annotate(qgraph)
        print('after annotate')
        print(qgraph.astext(show_meta_data=False))
        qgraph = qtz.calibrate(qgraph)
        print('after calibrate\n')
        print(qgraph.astext(show_meta_data=False))
        if not args.simulated:
            qgraph = qtz.realize(qgraph)
            qgraph = relay.ir_pass.infer_type(qgraph)
            print('after realize\n')
            print(qgraph.astext(show_meta_data=False))

    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(qgraph, target)
    ctx = tvm.nd.context(target, 0)
    return graph, lib, params, ctx


def main(args):
    gluon_model = vision.get_model(args.model, pretrained=True)
    #gluon_model = glcv2_get_model(args.model, pretrained=True)
    graph, lib, params, ctx = build_model(args, gluon_model)
    logging.info("Finish building model %s...", args.model)
    # raise ValueError
    evaluate(args, graph, lib, params, ctx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ImageNet validation accuracy")
    parser.add_argument("--rec-val", type=str, default="~/.mxnet/datasets/imagenet/rec/val.rec",
                        help="the validation data")
    parser.add_argument("--num-classes", type=int, default=1000,
                        help="batch size")
    parser.add_argument("--model", type=str, default="resnet50_v2",
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
    logging.basicConfig(level=logging.DEBUG)
    logging.info(args)
    main(args)
