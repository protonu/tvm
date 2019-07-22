import tvm
from tvm import relay

x = relay.var('x', shape=(10, 1))
y = relay.var('y', shape=(10, 1))
z = x + x * y

f = relay.Function([x, y], z)

print(f)

f = relay.ir_pass.infer_type(f)
fused_f = relay.ir_pass.fuse_ops(f, opt_level=3)

print(fused_f)

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
			new_op = self.rewrite_map[tuple(ops)]
			return relay.Call(new_op, call.args)
		else:
			return super().visit_call(call)

rewrite_map = {
	(relay.op.get('add'), relay.op.get('multiply')): relay.op.get("add")
	}

rop = RewriteQuantizedFake(rewrite_map).visit(fused_f)
rop = relay.ir_pass.infer_type(rop)
rop = relay.ir_pass.fuse_ops(rop)

print("after node rewrite")
print(rop)

