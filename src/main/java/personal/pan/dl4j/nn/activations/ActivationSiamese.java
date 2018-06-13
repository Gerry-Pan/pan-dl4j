package personal.pan.dl4j.nn.activations;

import org.nd4j.linalg.activations.BaseActivationFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.primitives.Pair;

public class ActivationSiamese extends BaseActivationFunction {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	private final double threshold;

	public ActivationSiamese(double threshold) {
		this.threshold = threshold;
	}

	@Override
	public INDArray getActivation(INDArray in, boolean training) {
		return siamese(in);
	}

	/*
	 * dLda=epsilon,在Siamese损失函数的反向传播时，此处的epsilon=labels
	 */
	@Override
	public Pair<INDArray, INDArray> backprop(INDArray in, INDArray epsilon) {
		int row = epsilon.rows();
		INDArray dLdz = Nd4j.create(row, 1);

		INDArray derivativeNegative = in.dup(in.ordering());

		BooleanIndexing.replaceWhere(derivativeNegative, 0, Conditions.greaterThanOrEqual(threshold));
		BooleanIndexing.replaceWhere(derivativeNegative, in.mul(2), Conditions.notEquals(0));

		INDArray derivativePositive = in.rsub(1).div(2);

		INDArray derivative = Nd4j.create(in.shape());

		derivative.put(new INDArrayIndex[] { NDArrayIndex.all(), NDArrayIndex.point(0) }, derivativeNegative);
		derivative.put(new INDArrayIndex[] { NDArrayIndex.all(), NDArrayIndex.point(1) }, derivativePositive);

		for (int i = 0; i < row; i++) {
			INDArray v1 = derivative.get(new INDArrayIndex[] { NDArrayIndex.point(i), NDArrayIndex.all() });
			INDArray v2 = epsilon.get(new INDArrayIndex[] { NDArrayIndex.point(i), NDArrayIndex.all() });

			dLdz.put(new INDArrayIndex[] { NDArrayIndex.point(i), NDArrayIndex.all() }, v1.mmul(v2.transpose()));
		}

		return new Pair<>(dLdz, null);
	}

	protected INDArray siamese(INDArray in) {
		INDArray outputPositive = in.rsub(1).div(2);
		outputPositive.muli(outputPositive);

		INDArray outputNegative = in.dup(in.ordering());

		BooleanIndexing.replaceWhere(outputNegative, 0, Conditions.greaterThanOrEqual(threshold));
		BooleanIndexing.replaceWhere(outputNegative, in.mul(in), Conditions.notEquals(0));

		INDArray output = Nd4j.create(in.rows(), in.columns() + 1);

		output.put(new INDArrayIndex[] { NDArrayIndex.all(), NDArrayIndex.point(0) }, outputNegative);
		output.put(new INDArrayIndex[] { NDArrayIndex.all(), NDArrayIndex.point(1) }, outputPositive);

		return output;
	}

}
