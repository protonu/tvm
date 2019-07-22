import math
import numpy as np
import tvm
from tvm import relay
from tvm.relay import quantize as qtz


##This code block is uses to rewrite a fused node, which is a primitive fn
rewrite_map = {
	(relay.op.get('add'), relay.op.get('multiply')): relay.op.get("add"),
	(relay.op.get('multiply'), relay.op.get('cast'),relay.op.get('nn.conv2d')): relay.op.get("nn.conv2d")
	}

class CollectAllOps(relay.expr_functor.ExprVisitor):
	def __init__(self):
		self.ops = []
		#super().__init__()
		relay.expr_functor.ExprVisitor.__init__(self)

	def visit_call(self, call):
		self.ops.append(call.op)
		#super().visit_call(call)
		relay.expr_functor.ExprVisitor.visit_call(self, call)

def get_all_ops(func):
	co = CollectAllOps()
	co.visit(func)
	return co.ops


class RewriteQuantizedFake(relay.expr_functor.ExprMutator):
	def __init__(self, rewrite_map):
		self.rewrite_map = rewrite_map
		#super(relay.expr_functor.ExprMutator, self).__init__()
		relay.expr_functor.ExprMutator.__init__(self)

	def visit_call(self, call):
		if isinstance(call.op, relay.Function) and call.op.attrs.Primitive.value == 1:
			ops = get_all_ops(call.op)
			if tuple(ops) in self.rewrite_map:
				new_op = self.rewrite_map[tuple(ops)]
				return relay.Call(new_op, call.args)
			else:
				return relay.expr_functor.ExprMutator.visit_call(self, call)
		else:
			#return super().visit_call(call)
			return relay.expr_functor.ExprMutator.visit_call(self, call)

def rewrite_node_in_graph(fused_graph):
    print("before node rewrite")
    print(fused_graph.astext(show_meta_data=False))

    rop = RewriteQuantizedFake(rewrite_map).visit(fused_graph)
    print("after node rewrite")
    print(rop.astext(show_meta_data=False))

    #This fails for conv
    #rop = relay.ir_pass.infer_type(rop)
    #print("after node rewrite and infer type pass")
    #print(rop.astext(show_meta_data=False))

    return rop

#End of code to rewrite fused node

def evaluate(graph):

    # tetup dataset.
    batch_size = 1

    # create runtime module
    # m = graph_runtime.create(graph, lib, ctx)
    target = "llvm"
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(graph, target)

    ctx = tvm.nd.context(target, 0)
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

def make_graph_conv2d(data):
    weight = relay.var("conv_weight")
    out = relay.nn.conv2d(data, weight, kernel_size=(3, 3), padding=(1, 1), channels=c)
    out = relay.Function(relay.ir_pass.free_vars(out), out)
    print("out")
    print(out.astext(show_meta_data=False))
    return out

def make_graph_fc():
    x = relay.var("x", shape=(10, 5))
    w = relay.var("w", shape=(2, 5))
    z = relay.nn.dense(x, w)
    func = relay.Function([x, w], z)
    func = relay.ir_pass.infer_type(func)
    return func

def make_graph_multiple_ops():
   # dshape = (1, 16, 64, 64)
    n, c, h, w = 1, 16, 64, 64
    data = relay.var("data", relay.TensorType((n, c, h, w), "float32"))
    #x = relay.var("x", shape=dshape)
    #x = relay.add(x, relay.const(1, "float32"))
    y = relay.nn.conv2d(data, relay.var("w1"),
                        kernel_size=(3, 3),
                        padding=(1, 1),
                        channels=16)
    # this is the next dominator.
    y = relay.add(relay.const(1, "float32"), y)
    #y = relay.add(y, y1)
    # second path
    z2 = relay.nn.conv2d(y, relay.var("w2"),
                         kernel_size=(1, 1),
                         padding=(0,0),
                         channels=16)
   # z3 = relay.nn.conv2d(y, relay.var("w3"),
   #                      kernel_size=(3, 3),
   #                      padding=(1,1),
   #                      channels=16)
    # add can only be fused to z1
    y = relay.add(y, z2)
    #return relay.Function(relay.ir_pass.free_vars(z2), z2)
    out = relay.Function(relay.ir_pass.free_vars(y), y)
    print("final y")
    print(out.astext(show_meta_data=False))
    return out


def make_dataset(graph, size=100):
	args = relay.ir_pass.infer_type(graph).params
        def create_arr(var):
            ttype = var.type_annotation
            np_arr = np.random.uniform(-1.0, 1.0, size=ttype.concrete_shape).astype(ttype.dtype)
            return tvm.ndarray.array(np_arr)

        params = {}
        dataset = []
        for arg in args:
            if arg.name_hint == 'data':
                dataset = [{'data': create_arr(arg)} for _ in range(size)]
            else:
                params[arg.name_hint] = create_arr(arg)
        return dataset, params

def quantization_method_a(graph, params):
    target = "llvm"	
    with relay.build_config(opt_level=3):
        graph = relay.optimize(graph, target, params)
    print("print this out")
    print(graph.astext(show_meta_data=False))

    with qtz.qconfig(skip_k_conv=0,
                     nbit_input=8,
                     nbit_weight=8,
                     global_scale=8.0,
                     dtype_input="uint8",
                     dtype_weight="int8",
                     dtype_activation="int32",
                     store_lowbit_output=False,
                     debug_enabled_ops=None):
        print(qtz.current_qconfig())
        graph = qtz.annotate(graph)
        print('after annotate')
        print(graph.astext(show_meta_data=False))
        graph = qtz.calibrate(graph)
        print('after calibrate\n')
        print(graph.astext(show_meta_data=False))
        graph = qtz.realize(graph)
        print('after realize\n')
        print(graph.astext(show_meta_data=False))
        graph = relay.ir_pass.infer_type(graph)
        return graph

def quantization_method_b(graph, params):
    with qtz.qconfig(skip_k_conv=0, global_scale=4.0,
                     round_for_shift=False, store_lowbit_output=False):
        graph = qtz.quantize(graph, params)
        graph = relay.ir_pass.infer_type(graph)
        return graph


if __name__ == "__main__":
    n, c, h, w = 1, 3, 224, 224
    data = relay.var("data", relay.TensorType((n, c, h, w), "float32"))
    qgraph = make_graph_conv2d(data)
    dataset, params = make_dataset(qgraph, 10)

    fc = make_graph_fc()
    dataset_fc, params_fc = make_dataset(fc, 10)
    print(fc.astext(show_meta_data=False))

    graph_conv_b = quantization_method_b(qgraph, params)
    graph_conv_a = quantization_method_a(qgraph, params)

    graph_fc_b = quantization_method_b(fc, params_fc)
    graph_fc_a = quantization_method_a(fc, params_fc)

    fused_graph_conv_b = relay.ir_pass.fuse_ops(graph_conv_b, opt_level=3)
    fused_graph_conv_a = relay.ir_pass.fuse_ops(graph_conv_a, opt_level=3)
    fused_graph_fc_a = relay.ir_pass.fuse_ops(graph_fc_a, opt_level=3)
    fused_graph_fc_b = relay.ir_pass.fuse_ops(graph_fc_b, opt_level=3)

    rewrite_node_in_graph(fused_graph_conv_a)
    #rewrite_node_in_graph(fused_graph_conv_b)

    multi_ops = make_graph_multiple_ops()
    print("multi op graph")
    print(multi_ops.astext(show_meta_data=False))
    dataset_ops, params_ops = make_dataset(multi_ops, 10)
    q_multi_ops = quantization_method_a(multi_ops, params_ops)
    print("quantized multi op graph")
    print(q_multi_ops.astext(show_meta_data=False))
    q_multi_ops = relay.ir_pass.fuse_ops(q_multi_ops)
    print("fused quantized multi op graph")
    print(q_multi_ops.astext(show_meta_data=False))

    evaluate(q_multi_ops)
