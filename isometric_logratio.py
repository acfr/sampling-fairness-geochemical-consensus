"""
MSF-GC. This module computes isometric log-ratio transformation
Copyright (C) 2022  Raymond Leung

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Further information about this program can be obtained from:
- Raymond Leung (raymond.leung@sydney.edu.au)
"""

import numpy
from scipy.linalg import norm, helmert
from scipy.stats.mstats import gmean


class IsometricLogRatio:
    """
    Implement isometric log-ratio transformation
    """

    def compute(self, x):
        """
        For a vector x of length m, corresponding to coordinates in Aitchison geometry,
        ilr(x) returns a (m-1,) vector of coordinates in Euclidian geometry.
        For a matrix x of size (n,m), it performs ilr(x[i,:]) on each row i,
        returning a (n,m-1) matrix.
        """
        x = numpy.array(x)
        n = x.shape[1]
        V = numpy.identity(n)
        V = V[:n,:n-1] - numpy.vstack((numpy.zeros(n-1),V[1:n,1:n]))
        U = self.gram_schmidt(V)[0]
        if numpy.isclose(norm(U,2), 1.0) == False:
            U = helmert(n).transpose()
        y = numpy.matmul(self.centered_log_ratio(x), U)
        return y

    @classmethod
    def gram_schmidt(cls, x):
        """
        Generates default orthonormal bases in the columns of Q
        """
        n = x.shape[1]
        R = numpy.zeros((n,n))
        Q = numpy.zeros((n+1,n))
        for j in range(n):
            v = numpy.array(x[:,j])
            for i in range(j):
                R[i,j] = numpy.dot(Q[:,i], x[:,j])
                v -= R[i,j] * Q[:,i]
            R[j,j] = norm(v)
            Q[:,j] = v / R[j,j]
        return Q, R

    @classmethod
    def centered_log_ratio(cls, x):
        """
        Perform centered log-ratio transform which maps compositional data
        from Aitchison Simplex to Euclidean vector space.
        """
        y = numpy.log(x / gmean(x,axis=1)[:, numpy.newaxis])
        return y

    def apply(self, features, valid_assays):
        x = features[valid_assays]
        #mask zero values
        smallest_non_zero = numpy.array([min([x[r,c] for r in range(x.shape[0]) if x[r,c] > 0])
                                         for c in range(x.shape[1])])
        x = numpy.maximum(x, smallest_non_zero)
        return self.compute(x)
