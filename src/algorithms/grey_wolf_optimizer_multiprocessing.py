from pathos.multiprocessing import ProcessPool
from numpy import fabs
from NiaPy.algorithms.basic import GreyWolfOptimizer


class GreyWolfOptimizerMultiprocessing(GreyWolfOptimizer):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

		self.nodes = kwargs.get('nodes', 4)

	def runIteration(self, task, pop, fpop, xb, fxb, A, A_f, B, B_f, D, D_f, **dparams):
		r"""Core funciton of GreyWolfOptimizer algorithm.
		Args:
			task (Task): Optimization task.
			pop (numpy.ndarray): Current population.
			fpop (numpy.ndarray): Current populations function/fitness values.
			xb (numpy.ndarray):
			fxb (float):
			A (numpy.ndarray):
			A_f (float):
			B (numpy.ndarray):
			B_f (float):
			D (numpy.ndarray):
			D_f (float):
			**dparams (Dict[str, Any]): Additional arguments.
		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, Dict[str, Any]]:
				1. New population
				2. New population fitness/function values
				3. Additional arguments:
					* A (): TODO
		"""

		def eval_task(args):
			i, w = args
			A1, C1 = 2 * a * self.rand(task.D) - a, 2 * self.rand(task.D)
			X1 = A - A1 * fabs(C1 * A - w)
			A2, C2 = 2 * a * self.rand(task.D) - a, 2 * self.rand(task.D)
			X2 = B - A2 * fabs(C2 * B - w)
			A3, C3 = 2 * a * self.rand(task.D) - a, 2 * self.rand(task.D)
			X3 = D - A3 * fabs(C3 * D - w)
			pop = task.repair((X1 + X2 + X3) / 3, self.Rand)
			fpop = task.eval(pop[i])
			return i, pop, fpop

		a = 2 - task.Evals * (2 / task.nFES)
		pool = ProcessPool(nodes=self.nodes)
		pool.clear()
		results = pool.map(eval_task, [[i, w] for i, w in enumerate(pop)])

		for i, _pop, _fpop in results:
			pop[i] = _pop
			fpop[i] = _fpop

		for i, f in enumerate(fpop):
			if f < A_f: A, A_f = pop[i].copy(), f
			elif A_f < f < B_f: B, B_f = pop[i].copy(), f
			elif B_f < f < D_f: D, D_f = pop[i].copy(), f
		xb, fxb = self.getBest(A, A_f, xb, fxb)
		return pop, fpop, xb, fxb, {'A': A, 'A_f': A_f, 'B': B, 'B_f': B_f, 'D': D, 'D_f': D_f}
