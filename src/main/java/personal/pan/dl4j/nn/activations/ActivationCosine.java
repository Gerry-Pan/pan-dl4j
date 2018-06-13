package personal.pan.dl4j.nn.activations;

import org.nd4j.linalg.activations.BaseActivationFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;

public class ActivationCosine extends BaseActivationFunction {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public ActivationCosine() {

	}

	@Override
	public INDArray getActivation(INDArray in, boolean training) {
		int columns = in.shape()[1];

		INDArray x1 = in.get(new INDArrayIndex[] { NDArrayIndex.all(), NDArrayIndex.interval(0, columns / 2) });
		INDArray x2 = in.get(new INDArrayIndex[] { NDArrayIndex.all(), NDArrayIndex.interval(columns / 2, columns) });

		INDArray x1mag = x1.norm2(1);
		INDArray x2mag = x2.norm2(1);
		x1mag = Transforms.max(x1mag, Nd4j.EPS_THRESHOLD, false);
		x2mag = Transforms.max(x2mag, Nd4j.EPS_THRESHOLD, false);

		INDArray preOut = x1.mul(x2);
		preOut.diviColumnVector(x1mag);
		preOut.diviColumnVector(x2mag);

		return preOut.sum(1);
	}

	/*
	 * dLda=epsilon,在Siamese损失函数的反向传播时，此处的epsilon=labels
	 */
	@Override
	public Pair<INDArray, INDArray> backprop(INDArray in, INDArray epsilon) {
		int columns = in.shape()[1];

		INDArray x1 = in.get(new INDArrayIndex[] { NDArrayIndex.all(), NDArrayIndex.interval(0, columns / 2) });
		INDArray x2 = in.get(new INDArrayIndex[] { NDArrayIndex.all(), NDArrayIndex.interval(columns / 2, columns) });

		INDArray x1L2norm = x1.norm2(1);
		INDArray x2L2norm = x2.norm2(1);

		INDArray x1L2normSq = x1L2norm.mul(x1L2norm);
		INDArray x1Dotx2L1norm = x2.mul(x1).sum(1);
		INDArray x1EpsilonNext = x2.mulColumnVector(x1L2normSq);
		x1EpsilonNext.subi(x1.mulColumnVector(x1Dotx2L1norm));

		INDArray x2L2normSq = x2L2norm.mul(x2L2norm);
		INDArray x2Dotx1L1norm = x1.mul(x2).sum(1);
		INDArray x2EpsilonNext = x1.mulColumnVector(x2L2normSq);
		x2EpsilonNext.subi(x2.mulColumnVector(x2Dotx1L1norm));

		x1L2norm = Transforms.max(x1L2norm, Nd4j.EPS_THRESHOLD, false);
		x2L2norm = Transforms.max(x2L2norm, Nd4j.EPS_THRESHOLD, false);
		x2L2normSq = Transforms.max(x2L2normSq, Nd4j.EPS_THRESHOLD, false);

		x1EpsilonNext.diviColumnVector(x2L2norm);
		x1EpsilonNext.diviColumnVector(x1L2norm.mul(x1L2normSq));

		x2EpsilonNext.diviColumnVector(x1L2norm);
		x2EpsilonNext.diviColumnVector(x2L2norm.mul(x2L2normSq));

		INDArray epsilonNext = Nd4j.hstack(x1EpsilonNext, x2EpsilonNext);

		epsilonNext.muliColumnVector(epsilon);

		return new Pair<>(epsilonNext, null);
	}

}
