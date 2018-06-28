package personal.pan.dl4j.nn.activations;

import org.nd4j.linalg.activations.BaseActivationFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.primitives.Pair;

public class ActivationL2Distance extends BaseActivationFunction {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public ActivationL2Distance() {

	}

	@Override
	public INDArray getActivation(INDArray in, boolean training) {
		int columns = in.shape()[1];

		INDArray x1 = in.get(new INDArrayIndex[] { NDArrayIndex.all(), NDArrayIndex.interval(0, columns / 2) });
		INDArray x2 = in.get(new INDArrayIndex[] { NDArrayIndex.all(), NDArrayIndex.interval(columns / 2, columns) });

		INDArray preOut = x1.sub(x2).norm2(1);

		return preOut;
	}

	/*
	 * dLda=epsilon,在Siamese损失函数的反向传播时，此处的epsilon=labels
	 */
	@Override
	public Pair<INDArray, INDArray> backprop(INDArray in, INDArray epsilon) {
		int columns = in.shape()[1];

		INDArray x1 = in.get(new INDArrayIndex[] { NDArrayIndex.all(), NDArrayIndex.interval(0, columns / 2) });
		INDArray x2 = in.get(new INDArrayIndex[] { NDArrayIndex.all(), NDArrayIndex.interval(columns / 2, columns) });

		INDArray g = x1.sub(x2);
		INDArray gL2norm = g.norm2(1);

		INDArray x1EpsilonNext = g.divColumnVector(gL2norm);
		INDArray x2EpsilonNext = g.mul(-1).divColumnVector(gL2norm);

		INDArray epsilonNext = Nd4j.hstack(x1EpsilonNext, x2EpsilonNext);

		epsilonNext.muliColumnVector(epsilon);

		return new Pair<>(epsilonNext, null);
	}

	@Override
	public String toString() {
		return "l2Distance";
	}
}
