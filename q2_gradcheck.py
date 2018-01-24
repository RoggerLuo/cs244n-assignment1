#!/usr/bin/env python
import numpy as np
import random
from q2_sigmoid import sigmoid 
# First implement a gradient checker by filling in the following functions
def gradcheck_naive(f, x):
    """ Gradient check for a function f.
    Arguments:
    f -- a function that takes a single argument and outputs the
         cost and its gradients
    x -- the point (numpy array) to check the gradient at
    """

    rndstate = random.getstate()
    random.setstate(rndstate)
    fx, grad = f(x) # Evaluate function value at original point
    h = 1e-4        # Do not change this!

    # Iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        # Try modifying x[ix] with h defined above to compute
        # numerical gradients. Make sure you call random.setstate(rndstate)
        # before calling f(x) each time. This will make it possible
        # to test cost functions with built in randomness later.

        ### YOUR CODE HERE:
        
        old_value = x[ix]

        random.setstate(rndstate)
        fx2, grad2 = f(x)

        #右边
        x[ix] = old_value + h
        random.setstate(rndstate)
        fxh_left, aaa = f(x)
        # print('old_value + h')
        # print(x[ix])
        # print('old_value + h: cost cost cost cost cost')
        # print(fxh_left)

        # print('left cost')
        # print(fxh_left)

        #左边
        x[ix] = old_value - h
        random.setstate(rndstate)
        fxh_right,aaa = f(x)
        # print('right cost')
        # print(fxh_right)

        # 还原
        x[ix] = old_value 

        numgrad = (fxh_left - fxh_right) / (2*h) 
        

        # 这个对于多位数组貌似挺重要，不然sigmoid就通不过
        numgrad = np.sum(numgrad)
        ### END YOUR CODE

        # Compare gradients
        reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))
        if reldiff > 1e-5:
            print("Gradient check failed.")
            print("First gradient error found at index %s" % str(ix))
            print("Your gradient: %f \t Numerical gradient: %f" % (
                grad[ix], numgrad) )
            return
        # print("Your gradient: %f \t Numerical gradient: %f" % (
        #     grad[ix], numgrad) )

        it.iternext() # Step to next dimension

    print("Gradient check passed!")


def sanity_check():
    """
    Some basic sanity checks.
    """
    # 左边是cost，右边是grad 
    quad = lambda x: (np.sum(x ** 2), x * 2.0000001)

    print("Running sanity checks...")
    gradcheck_naive(quad, np.array(123.456))      # scalar test
    gradcheck_naive(quad, np.random.randn(3,))    # 1-D test
    gradcheck_naive(quad, np.random.randn(4,5))   # 2-D test


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_gradcheck.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print("Running your sanity checks...")
    ### YOUR CODE HERE
    quad = lambda x:( sigmoid(x), sigmoid(x)*(1-sigmoid(x)))
    gradcheck_naive(quad, np.array(123.456))      # scalar test
    gradcheck_naive(quad, np.random.randn(4,5))   # 2-D test

    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
