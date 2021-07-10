# from sdtw import SoftDTW
# from sdtw.distance import SquaredEuclidean
#
# # Time series 1: numpy array, shape = [m, d] where m = length and d = dim
# X = ...
# # Time series 2: numpy array, shape = [n, d] where n = length and d = dim
# Y = ...
#
# # D can also be an arbitrary distance matrix: numpy array, shape [m, n]
# D = SquaredEuclidean(X, Y)
# sdtw = SoftDTW(D, gamma=1.0)
# # soft-DTW discrepancy, approaches DTW as gamma -> 0
# value = sdtw.compute()
# # gradient w.r.t. D, shape = [m, n], which is also the expected alignment matrix
# E = sdtw.grad()
# # gradient w.r.t. X, shape = [m, d]
# G = D.jacobian_product(E)