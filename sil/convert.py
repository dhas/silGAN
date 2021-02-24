import ast
import astor
import copy

class CompareTransformer(ast.NodeTransformer):
	def visit_Compare(self, node):
		calls = []
		for i, op in enumerate(node.ops):
			if i == 0:
				left_op = node.left
			else:
				left_op = node.comparators[i-1]
			right_op = node.comparators[i]
			if isinstance(left_op, ast.Num):
				left_op = ast.Call(ast.Name('loss.const', ast.Load()), [left_op], [])
			if isinstance(right_op, ast.Num):
				right_op = ast.Call(ast.Name('loss.const', ast.Load()), [right_op], [])

			if isinstance(op, ast.LtE):
				call = ast.Call(ast.Name('loss.b2f_LT', ast.Load()), [left_op, right_op], [])
				calls.append(call)
		if len(calls) > 1:
			node = ast.Call(ast.Name('loss.b2f_AND', ast.Load()), calls, [])
		else:
			node = calls[0]
		return node
	
class BranchTransformer(ast.NodeTransformer):
	def __init__(self):
		super().__init__()
		self.branch_count = 0

	def visit_Return(self, node):
		loss_elts = [ast.Name(id='c%d' % count, ctx=ast.Store()) for count in range(self.branch_count)]
		loss_list = ast.List(elts=loss_elts, ctx=ast.Load())
		loss_stack = ast.Call(ast.Name('torch.stack', ast.Load()), [loss_list], [])

		hit_elts = [ast.Name(id='h%d' % count, ctx=ast.Store()) for count in range(self.branch_count)]
		hits_list = ast.List(elts=hit_elts, ctx=ast.Load())
		hits_stack = ast.Call(ast.Name('torch.stack', ast.Load()), [hits_list], [])

		ret_tuple = ast.Tuple(elts=[loss_stack, hits_stack], ctx=ast.Load())

		node = ast.Return(value=ret_tuple)
		self.branch_count = 0
		return node

	def loss_to_hit(self, node):
		_cnode = copy.deepcopy(node)
		for n in ast.walk(_cnode):
			if isinstance(n, ast.Call):
				if n.func.id == 'loss.b2f_AND':
					n.func.id = 'torch.logical_and'
				elif n.func.id == 'loss.b2f_LT':
					n.func.id = 'torch.lt'
		return _cnode

	def visit_If(self, node):
		loss_node = ast.Assign(targets=[ast.Name(id='c%d' % self.branch_count, ctx=ast.Store())], value=node.test)
		test_copy = self.loss_to_hit(node.test)
		hit_node  = ast.Assign(targets=[ast.Name(id='h%d' % self.branch_count, ctx=ast.Store())], value=test_copy)		
		node = ast.Module(
			body=[loss_node, hit_node])
		self.branch_count +=1
		return node


if __name__ == '__main__':	
	with open('test.py') as source:
		tree = ast.parse(source.read())
		ast.fix_missing_locations(CompareTransformer().visit(tree))
		ast.fix_missing_locations(BranchTransformer().visit(tree))
		import_node = ast.Import(names=[ast.alias(name='loss', asname=None)])
		tree.body.insert(2, import_node)
		new_source = astor.to_source(tree)
		with open('search.py', 'w') as dest:
			dest.write('#This file has been generated\n')
			dest.write(new_source)
		# for node in ast.walk(tree):
		# 	if isinstance(node, ast.LtE):
		# 		break