import queue as q
class ReversePriorityQueue(q.PriorityQueue):
	def put(self, tup):
		newtup = (tup[0] * -1,)+ tup[1:]
		PriorityQueue.put(self, newtup)

	def get(self):
		tup = PriorityQueue.get(self)
		newtup = (tup[0] * -1,)+ tup[1:]
		return newtup


q= ReversePriorityQueue()
q.put((-0.09,12,14))
q.put((-0.1,14,17))
q.put((0,1,1))
print(q.get())