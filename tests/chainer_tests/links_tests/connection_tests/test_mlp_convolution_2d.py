import unittest

import numpy

import chainer
from chainer import functions
from chainer import links
from chainer import testing


@testing.parameterize(
    {'use_cudnn': True},
    {'use_cudnn': False},
)
class TestMLPConvolution2D(unittest.TestCase):

    def setUp(self):
        self.mlp = links.MLPConvolution2D(
            3, (96, 96, 96), 11,
            activation=functions.sigmoid,
            use_cudnn=self.use_cudnn)

    def test_init(self):
        self.assertIs(self.mlp.activation, functions.sigmoid)

        self.assertEqual(len(self.mlp), 3)
        for i, conv in enumerate(self.mlp):
            self.assertIsInstance(conv, links.Convolution2D)
            self.assertEqual(conv.use_cudnn, self.use_cudnn)
            if i == 0:
                self.assertEqual(conv.W.data.shape, (96, 3, 11, 11))
            else:
                self.assertEqual(conv.W.data.shape, (96, 96, 1, 1))

    def test_call(self):
        x = chainer.Variable(numpy.zeros((10, 3, 10, 10), dtype=numpy.float32))
        actual = self.mlp(x)
        act = functions.sigmoid
        expect = self.mlp[2](act(self.mlp[1](act(self.mlp[0](x)))))
        numpy.testing.assert_equal(expect.data, actual.data)


testing.run_module(__name__, __file__)
