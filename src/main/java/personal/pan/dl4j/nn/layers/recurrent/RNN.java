package personal.pan.dl4j.nn.layers.recurrent;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.recurrent.BaseRecurrentLayer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

/**
 * 原始的RNN算法<br>
 * created on 2018-1-3
 * 
 * @author Gerry Pan
 *
 */
public class RNN extends BaseRecurrentLayer<personal.pan.dl4j.nn.conf.layers.RNN> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public static final String STATE_KEY_PREV_ACTIVATION = "prevAct";

	public RNN(NeuralNetConfiguration conf) {
		super(conf);
	}

	@Override
	public INDArray rnnTimeStep(INDArray input) {
		return null;
	}

	@Override
	public INDArray rnnActivateUsingStoredState(INDArray input, boolean training, boolean storeLastForTBPTT) {
		return null;
	}

	@Override
	public Gradient gradient() {
		throw new UnsupportedOperationException(
				"gradient() method for layerwise pretraining: not supported for RNN (pretraining not possible)"
						+ layerId());
	}

	@Override
	public Pair<Gradient, INDArray> tbpttBackpropGradient(INDArray epsilon, int tbpttBackwardLength) {
		return backpropGradientHelper(epsilon, true, tbpttBackwardLength);
	}

	@Override
	public boolean isPretrainLayer() {
		return false;
	}

	@Override
	public Gradient calcGradient(Gradient layerError, INDArray activation) {
		throw new UnsupportedOperationException("Not supported " + layerId());
	}

	@Override
	public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon) {
		return backpropGradientHelper(epsilon, false, -1);
	}

	protected Pair<Gradient, INDArray> backpropGradientHelper(final INDArray epsilon, final boolean truncatedBPTT,
			final int tbpttBackwardLength) {
		// 即将实现，敬请期待
		return null;
	}

	@Override
	public INDArray preOutput(INDArray x) {
		return activate(x, true);
	}

	@Override
	public INDArray preOutput(INDArray x, boolean training) {
		return activate(x, training);
	}

	@Override
	public INDArray activate(INDArray input, boolean training) {
		setInput(input);
		return activateHelper(training, null, null, false).fwdPassOutput;
	}

	@Override
	public INDArray activate(INDArray input) {
		setInput(input);
		return activateHelper(true, null, null, false).fwdPassOutput;
	}

	@Override
	public INDArray activate(boolean training) {
		return activateHelper(training, null, null, false).fwdPassOutput;
	}

	@Override
	public INDArray activate() {
		return activateHelper(false, null, null, false).fwdPassOutput;
	}

	protected RNNFwdPassReturn activateHelper(final boolean training, final INDArray prevOutputActivations,
			final INDArray prevMemCellState, boolean forBackprop) {
		// 即将实现，敬请期待
		return null;
	}
}
