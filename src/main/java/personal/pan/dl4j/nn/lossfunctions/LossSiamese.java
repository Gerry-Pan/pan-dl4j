package personal.pan.dl4j.nn.lossfunctions;

import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.lossfunctions.LossUtil;
import org.nd4j.linalg.lossfunctions.impl.LossMCXENT;

import personal.pan.dl4j.nn.activations.ActivationCosine;
import personal.pan.dl4j.nn.activations.ActivationSiamese;

/**
 * https://blog.csdn.net/thriving_fcl/article/details/73730552
 * 
 * @author Jerry
 *
 */
public class LossSiamese extends LossMCXENT {

	private final double threshold;

	public LossSiamese(double threshold) {
		this.threshold = threshold;
	}

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	/**
	 * preOutput=[X1,X2]<br>
	 * output=cosine(preOutput)<br>
	 * if output < threshold,l=[output^2,(1-output)^2/4] <br>
	 * if output > threshold,l=[0,(1-output)^2/4]<br>
	 * l=[L−(X1,X2),L+(X1,X2)]<br>
	 * labels=[1-y,y]<br>
	 * L=Lw(X1,X2)=(1−y)L−(X1,X2)+yL+(X1,X2)<br>
	 * L=l.mul(labels)<br>
	 * 
	 * @param labels
	 *            shape(batchSize,2)
	 * @param preOutput
	 *            shape(batchSize,1)&nbsp;&nbsp;&nbsp;&nbsp;此参数为链接中的Ew,即cosine值
	 * @param activationFn
	 *            {@link ActivationSiamese}
	 * @param mask
	 * @return
	 */
	private INDArray scoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
		if (!(activationFn instanceof ActivationCosine)) {
			throw new RuntimeException("activation function must be ActivationSiamese.");
		}

		INDArray output = activationFn.getActivation(preOutput, true);

		INDArray outputPositive = output.rsub(1).div(2);
		outputPositive.muli(outputPositive);

		INDArray outputNegative = output.dup(output.ordering());

		BooleanIndexing.replaceWhere(outputNegative, 0, Conditions.greaterThanOrEqual(threshold));
		BooleanIndexing.replaceWhere(outputNegative, output.mul(output), Conditions.notEquals(0));

		INDArray l = Nd4j.create(labels.shape());

		l.put(new INDArrayIndex[] { NDArrayIndex.all(), NDArrayIndex.point(0) }, outputNegative);
		l.put(new INDArrayIndex[] { NDArrayIndex.all(), NDArrayIndex.point(1) }, outputPositive);

		INDArray scoreArr = l.mul(labels);

		if (mask != null) {
			LossUtil.applyMask(scoreArr, mask);
		}

		return scoreArr;
	}

	@Override
	public double computeScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask,
			boolean average) {
		INDArray scoreArr = scoreArray(labels, preOutput, activationFn, mask);

		double score = scoreArr.sumNumber().doubleValue();

		if (average) {
			score /= scoreArr.size(0);
		}

		return score;
	}

	@Override
	public INDArray computeScoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
		INDArray scoreArr = scoreArray(labels, preOutput, activationFn, mask);
		return scoreArr.sum(1);
	}

	@Override
	public INDArray computeGradient(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {

		if (!(activationFn instanceof ActivationCosine)) {
			throw new RuntimeException("activation function must be ActivationSiamese.");
		}

		INDArray dLdl = labels.dup();

		int row = dLdl.rows();
		INDArray dLda = Nd4j.create(row, 1);

		INDArray output = activationFn.getActivation(preOutput, true);

		INDArray derivativeNegative = output.dup(output.ordering());

		BooleanIndexing.replaceWhere(derivativeNegative, 0, Conditions.greaterThanOrEqual(threshold));
		BooleanIndexing.replaceWhere(derivativeNegative, output.mul(2), Conditions.notEquals(0));

		INDArray derivativePositive = output.rsub(1).div(2);

		INDArray derivative = Nd4j.create(output.rows(), output.columns() + 1);

		derivative.put(new INDArrayIndex[] { NDArrayIndex.all(), NDArrayIndex.point(0) }, derivativeNegative);
		derivative.put(new INDArrayIndex[] { NDArrayIndex.all(), NDArrayIndex.point(1) }, derivativePositive);

		for (int i = 0; i < row; i++) {
			INDArray v1 = derivative.get(new INDArrayIndex[] { NDArrayIndex.point(i), NDArrayIndex.all() });
			INDArray v2 = dLdl.get(new INDArrayIndex[] { NDArrayIndex.point(i), NDArrayIndex.all() });

			dLda.put(new INDArrayIndex[] { NDArrayIndex.point(i), NDArrayIndex.all() }, v1.mmul(v2.transpose()));
		}

		INDArray dLdoutput = activationFn.backprop(preOutput, dLda).getFirst();

		return dLdoutput;
	}
}
