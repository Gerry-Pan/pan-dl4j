package personal.pan.dl4j.nn.lossfunctions;

import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.LossUtil;
import org.nd4j.linalg.lossfunctions.impl.LossMCXENT;

import personal.pan.dl4j.nn.activations.ActivationSiamese;

/**
 * https://blog.csdn.net/thriving_fcl/article/details/73730552
 * 
 * @author Jerry
 *
 */
public class LossSiamese extends LossMCXENT {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	/**
	 * for (int i = 0; i < row; i++) {<br>
	 * &nbsp;&nbsp;&nbsp;&nbsp;double ew = output.getDouble(i, 0);<br>
	 * &nbsp;&nbsp;&nbsp;&nbsp;yhatNegative.putScalar(i, 0, ew < threshold ? ew * ew
	 * : 0);<br>
	 * }<br>
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
		if (labels.size(1) != (preOutput.size(1) + 1)) {
			throw new IllegalArgumentException("Labels array numColumns (size(1) = " + String.valueOf(labels.size(1))
					+ ") does not match output layer" + " number of outputs (nOut = "
					+ String.valueOf(preOutput.size(1) + 1) + ") ");

		}

		if (!(activationFn instanceof ActivationSiamese)) {
			throw new RuntimeException("activation function must be ActivationSiamese.");
		}

		INDArray output = activationFn.getActivation(preOutput.dup(), true);

		INDArray scoreArr = output.mul(labels);

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

	/*
	 * @param preOutput
	 *            shape(batchSize,1)&nbsp;&nbsp;&nbsp;&nbsp;此参数为链接中的Ew,即cosine值
	 */
	@Override
	public INDArray computeGradient(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {

		if (!(activationFn instanceof ActivationSiamese)) {
			throw new RuntimeException("activation function must be ActivationSiamese.");
		}

		INDArray dLda = labels.dup();
		INDArray gradients = activationFn.backprop(preOutput, dLda).getFirst();

		return gradients;
	}
}
