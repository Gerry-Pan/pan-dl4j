package personal.pan.dl4j.nn.layers.recurrent;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.recurrent.BaseRecurrentLayer;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

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

	public GRU(NeuralNetConfiguration conf, DataType dataType) {
		super(conf, dataType);
	}

	public boolean isPretrainLayer() {
		return false;
	}

	public INDArray rnnTimeStep(INDArray input, LayerWorkspaceMgr workspaceMgr) {
		setInput(input, workspaceMgr);
		GRUFwdPassReturn fwdPass = activateHelper(false, stateMap.get(STATE_KEY_PREV_ACTIVATION), false, workspaceMgr);
		INDArray outAct = fwdPass.fwdPassOutput;

		stateMap.put(STATE_KEY_PREV_ACTIVATION, fwdPass.lastAct);

		return outAct;
	}

	public INDArray rnnActivateUsingStoredState(INDArray input, boolean training, boolean storeLastForTBPTT,
			LayerWorkspaceMgr workspaceMgr) {
		setInput(input, workspaceMgr);
		GRUFwdPassReturn fwdPass = activateHelper(false, stateMap.get(STATE_KEY_PREV_ACTIVATION), false, workspaceMgr);
		INDArray outAct = fwdPass.fwdPassOutput;

		if (storeLastForTBPTT) {
			tBpttStateMap.put(STATE_KEY_PREV_ACTIVATION, fwdPass.lastAct.detach());
		}

		return outAct;
	}

	public Gradient gradient() {
		throw new UnsupportedOperationException(
				"gradient() method for layerwise pretraining: not supported for GRU (pretraining not possible)"
						+ layerId());
	}

	public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
		return backpropGradientHelper(epsilon, false, -1, workspaceMgr);
	}

	public Pair<Gradient, INDArray> tbpttBackpropGradient(INDArray epsilon, int tbpttBackwardLength,
			LayerWorkspaceMgr workspaceMgr) {
		return backpropGradientHelper(epsilon, true, tbpttBackwardLength, workspaceMgr);
	}

	protected Pair<Gradient, INDArray> backpropGradientHelper(final INDArray epsilon, final boolean truncatedBPTT,
			final int tbpttBackwardLength, LayerWorkspaceMgr workspaceMgr) {
		final INDArray inputWeights = getParam(GRUParamInitializer.INPUT_WEIGHT_KEY);
		final INDArray recurrentWeights = getParam(GRUParamInitializer.RECURRENT_WEIGHT_KEY);

		GRUFwdPassReturn fwdPass = null;

		if (truncatedBPTT) {
			fwdPass = activateHelper(true, stateMap.get(STATE_KEY_PREV_ACTIVATION), true, workspaceMgr);
			tBpttStateMap.put(STATE_KEY_PREV_ACTIVATION, fwdPass.lastAct.detach());
		} else {
			fwdPass = activateHelper(true, null, true, workspaceMgr);
		}

		return GRUHelpers.backpropGradientHelper(this.conf, this.layerConf().getGateActivationFn(), this.input,
				recurrentWeights, inputWeights, epsilon, truncatedBPTT, tbpttBackwardLength, fwdPass,
				GRUParamInitializer.INPUT_WEIGHT_KEY, GRUParamInitializer.RECURRENT_WEIGHT_KEY,
				GRUParamInitializer.BIAS_KEY, gradientViews, maskArray, workspaceMgr, getDataType());
	}

	public INDArray preOutput(INDArray x, LayerWorkspaceMgr workspaceMgr) {
		return activate(x, true, workspaceMgr);
	}

	public INDArray preOutput(INDArray x, boolean training, LayerWorkspaceMgr workspaceMgr) {
		return activate(x, training, workspaceMgr);
	}

	public INDArray activate(INDArray input, boolean training, LayerWorkspaceMgr workspaceMgr) {
		setInput(input, workspaceMgr);
		return activateHelper(training, null, false, workspaceMgr).fwdPassOutput;
	}

	public INDArray activate(INDArray input, LayerWorkspaceMgr workspaceMgr) {
		setInput(input, workspaceMgr);
		return activateHelper(true, null, false, workspaceMgr).fwdPassOutput;
	}

	public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {
		return activateHelper(training, null, false, workspaceMgr).fwdPassOutput;
	}

	public INDArray activate(LayerWorkspaceMgr workspaceMgr) {
		return activateHelper(false, null, false, workspaceMgr).fwdPassOutput;
	}

	protected GRUFwdPassReturn activateHelper(final boolean training, final INDArray prevOutputActivations,
			boolean forBackprop, LayerWorkspaceMgr workspaceMgr) {
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
				maskArray, forBackprop ? cacheMode : CacheMode.NONE, workspaceMgr, getDataType());

		if (training && cacheMode != CacheMode.NONE) {
			cachedFwdPass = fwd;
		}

		return fwd;
	}

	public Pair<INDArray, MaskState> feedForwardMaskArray(INDArray maskArray, MaskState currentMaskState,
			int minibatchSize) {
		return new Pair<>(maskArray, MaskState.Passthrough);
	}

	public Type type() {
		return Type.RECURRENT;
	}

	public Layer transpose() {
		throw new UnsupportedOperationException("Not supported " + layerId());
	}
}
