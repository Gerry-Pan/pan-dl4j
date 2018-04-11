package personal.pan.dl4j.nn.layers.recurrent;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.recurrent.BaseRecurrentLayer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

import personal.pan.dl4j.nn.params.GRUParamInitializer;

/**
 * created on 2017-12-29
 * 
 * @author Gerry Pan
 *
 */
public class GRU extends BaseRecurrentLayer<personal.pan.dl4j.nn.conf.layers.GRU> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public static final String STATE_KEY_PREV_ACTIVATION = "prevAct";

	protected GRUFwdPassReturn cachedFwdPass;

	public GRU(NeuralNetConfiguration conf) {
		super(conf);
	}

	@Override
	public boolean isPretrainLayer() {
		return false;
	}

	@Override
	public INDArray rnnTimeStep(INDArray input) {
		setInput(input);
		GRUFwdPassReturn fwdPass = activateHelper(false, stateMap.get(STATE_KEY_PREV_ACTIVATION), false);
		INDArray outAct = fwdPass.fwdPassOutput;

		stateMap.put(STATE_KEY_PREV_ACTIVATION, fwdPass.lastAct);

		return outAct;
	}

	@Override
	public INDArray rnnActivateUsingStoredState(INDArray input, boolean training, boolean storeLastForTBPTT) {
		setInput(input);
		GRUFwdPassReturn fwdPass = activateHelper(false, stateMap.get(STATE_KEY_PREV_ACTIVATION), false);
		INDArray outAct = fwdPass.fwdPassOutput;

		if (storeLastForTBPTT) {
			tBpttStateMap.put(STATE_KEY_PREV_ACTIVATION, fwdPass.lastAct.leverageTo(ComputationGraph.WORKSPACE_TBPTT));
		}

		return outAct;
	}

	@Override
	public Gradient gradient() {
		throw new UnsupportedOperationException(
				"gradient() method for layerwise pretraining: not supported for GRU (pretraining not possible)"
						+ layerId());
	}

	@Override
	public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon) {
		return backpropGradientHelper(epsilon, false, -1);
	}

	@Override
	public Pair<Gradient, INDArray> tbpttBackpropGradient(INDArray epsilon, int tbpttBackwardLength) {
		return backpropGradientHelper(epsilon, true, tbpttBackwardLength);
	}

	protected Pair<Gradient, INDArray> backpropGradientHelper(final INDArray epsilon, final boolean truncatedBPTT,
			final int tbpttBackwardLength) {
		final INDArray inputWeights = getParam(GRUParamInitializer.INPUT_WEIGHT_KEY);
		final INDArray recurrentWeights = getParam(GRUParamInitializer.RECURRENT_WEIGHT_KEY);

		GRUFwdPassReturn fwdPass = null;

		if (truncatedBPTT) {
			fwdPass = activateHelper(true, stateMap.get(STATE_KEY_PREV_ACTIVATION), true);
			tBpttStateMap.put(STATE_KEY_PREV_ACTIVATION, fwdPass.lastAct.leverageTo(ComputationGraph.WORKSPACE_TBPTT));
		} else {
			fwdPass = activateHelper(true, null, true);
		}

		return GRUHelpers.backpropGradientHelper(this.conf, this.layerConf().getGateActivationFn(), this.input,
				recurrentWeights, inputWeights, epsilon, truncatedBPTT, tbpttBackwardLength, fwdPass,
				GRUParamInitializer.INPUT_WEIGHT_KEY, GRUParamInitializer.RECURRENT_WEIGHT_KEY,
				GRUParamInitializer.BIAS_KEY, gradientViews, maskArray);
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
		return activateHelper(training, null, false).fwdPassOutput;
	}

	@Override
	public INDArray activate(INDArray input) {
		setInput(input);
		return activateHelper(true, null, false).fwdPassOutput;
	}

	@Override
	public INDArray activate(boolean training) {
		return activateHelper(training, null, false).fwdPassOutput;
	}

	@Override
	public INDArray activate() {
		return activateHelper(false, null, false).fwdPassOutput;
	}

	protected GRUFwdPassReturn activateHelper(final boolean training, final INDArray prevOutputActivations,
			boolean forBackprop) {
		if (cacheMode == null)
			cacheMode = CacheMode.NONE;

		if (forBackprop && cachedFwdPass != null) {
			GRUFwdPassReturn ret = cachedFwdPass;
			cachedFwdPass = null;
			return ret;
		}

		final INDArray recurrentWeights = getParam(GRUParamInitializer.RECURRENT_WEIGHT_KEY);
		final INDArray inputWeights = getParam(GRUParamInitializer.INPUT_WEIGHT_KEY);
		final INDArray biases = getParam(GRUParamInitializer.BIAS_KEY);

		GRUFwdPassReturn fwd = GRUHelpers.activateHelper(this, this.conf, this.layerConf().getGateActivationFn(),
				this.input, recurrentWeights, inputWeights, biases, training, prevOutputActivations,
				forBackprop || (cacheMode != CacheMode.NONE && training), GRUParamInitializer.INPUT_WEIGHT_KEY,
				maskArray, forBackprop ? cacheMode : CacheMode.NONE);

		if (training && cacheMode != CacheMode.NONE) {
			cachedFwdPass = fwd;
		}

		return fwd;
	}

	@Override
	public Type type() {
		return Type.RECURRENT;
	}

	@Override
	public Layer transpose() {
		throw new UnsupportedOperationException("Not supported " + layerId());
	}
}
