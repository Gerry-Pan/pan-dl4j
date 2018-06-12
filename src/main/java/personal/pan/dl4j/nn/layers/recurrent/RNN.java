package personal.pan.dl4j.nn.layers.recurrent;

import java.util.Arrays;
import java.util.Map;

import org.deeplearning4j.exception.DL4JInvalidInputException;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.recurrent.BaseRecurrentLayer;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.blas.Level1;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import personal.pan.dl4j.nn.params.RNNParamInitializer;

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

	private RNNFwdPassReturn cachedFwdPass;

	public RNN(NeuralNetConfiguration conf) {
		super(conf);
	}

	@Override
	public INDArray rnnTimeStep(INDArray input) {
		setInput(input);
		RNNFwdPassReturn fwdPass = activateHelper(false, stateMap.get(STATE_KEY_PREV_ACTIVATION), false);
		INDArray outAct = fwdPass.fwdPassOutput;

		stateMap.put(STATE_KEY_PREV_ACTIVATION, fwdPass.lastAct);

		return outAct;
	}

	@Override
	public INDArray rnnActivateUsingStoredState(INDArray input, boolean training, boolean storeLastForTBPTT) {
		setInput(input);
		RNNFwdPassReturn fwdPass = activateHelper(false, stateMap.get(STATE_KEY_PREV_ACTIVATION), false);
		INDArray outAct = fwdPass.fwdPassOutput;

		if (storeLastForTBPTT) {
			tBpttStateMap.put(STATE_KEY_PREV_ACTIVATION, fwdPass.lastAct.leverageTo(ComputationGraph.WORKSPACE_TBPTT));
		}

		return outAct;
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
	public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon) {
		return backpropGradientHelper(epsilon, false, -1);
	}

	protected Pair<Gradient, INDArray> backpropGradientHelper(final INDArray epsilon, final boolean truncatedBPTT,
			final int tbpttBackwardLength) {
		final INDArray inputWeights = getParam(RNNParamInitializer.INPUT_WEIGHT_KEY);
		final INDArray recurrentWeights = getParam(RNNParamInitializer.RECURRENT_WEIGHT_KEY);

		RNNFwdPassReturn fwdPass = null;

		if (truncatedBPTT) {
			fwdPass = activateHelper(true, stateMap.get(STATE_KEY_PREV_ACTIVATION), true);
			tBpttStateMap.put(STATE_KEY_PREV_ACTIVATION, fwdPass.lastAct.leverageTo(ComputationGraph.WORKSPACE_TBPTT));
		} else {
			fwdPass = activateHelper(true, null, true);
		}

		return backpropGradientHelper(this.conf, this.input, recurrentWeights, inputWeights, epsilon, truncatedBPTT,
				tbpttBackwardLength, fwdPass, RNNParamInitializer.INPUT_WEIGHT_KEY,
				RNNParamInitializer.RECURRENT_WEIGHT_KEY, RNNParamInitializer.BIAS_KEY, gradientViews, maskArray);
	}

	private static Pair<Gradient, INDArray> backpropGradientHelper(final NeuralNetConfiguration conf,
			final INDArray input, final INDArray recurrentWeights, final INDArray inputWeights, final INDArray epsilon,
			final boolean truncatedBPTT, final int tbpttBackwardLength, final RNNFwdPassReturn fwdPass,
			final String inputWeightKey, final String recurrentWeightKey, final String biasWeightKey,
			final Map<String, INDArray> gradientViews, INDArray maskArray) {
		int miniBatchSize = epsilon.size(0);
		boolean is2dInput = epsilon.rank() < 3;
		int prevLayerSize = inputWeights.size(0);// prevLayerSize=nLast
		int hiddenLayerSize = recurrentWeights.size(0);
		int timeSeriesLength = (is2dInput ? 1 : epsilon.size(2));

		int endIdx = 0;

		if (truncatedBPTT) {
			endIdx = Math.max(0, timeSeriesLength - tbpttBackwardLength);
		}

		INDArray dLdoNext = null;
		INDArray deltaNext = Nd4j.create(new int[] { miniBatchSize, hiddenLayerSize }, 'f');

		INDArray timeStepMaskColumn = null;
		INDArray epsilonNext = Nd4j.create(new int[] { miniBatchSize, prevLayerSize, timeSeriesLength }, 'f');
		IActivation afn = ((org.deeplearning4j.nn.conf.layers.BaseLayer) conf.getLayer()).getActivationFn();

		INDArray iwGradientsOut = gradientViews.get(inputWeightKey);// shape(nLast,3*hiddenLayerSize)
		INDArray rwGradientsOut = gradientViews.get(recurrentWeightKey);// shape(hiddenLayerSize,3*hiddenLayerSize)
		INDArray bGradientsOut = gradientViews.get(biasWeightKey);
		iwGradientsOut.assign(0);
		rwGradientsOut.assign(0);
		bGradientsOut.assign(0);

		Level1 l1BLAS = Nd4j.getBlasWrapper().level1();

		for (int iTimeIndex = timeSeriesLength - 1; iTimeIndex >= endIdx; iTimeIndex--) {
			int time = iTimeIndex;
			int inext = 1;

			INDArray epsilonSlice = (is2dInput ? epsilon : epsilon.tensorAlongDimension(time, 1, 0));

			if (iTimeIndex == timeSeriesLength - 1) {
				dLdoNext = Nd4j.create(new int[] { miniBatchSize, hiddenLayerSize }, 'f');
			}

			dLdoNext.addi(Shape.toOffsetZeroCopy(epsilonSlice, 'f'));

			Nd4j.gemm(deltaNext, recurrentWeights, dLdoNext, false, true, 1.0, 1.0);

			if (iTimeIndex != timeSeriesLength - 1) {
				INDArray oz = fwdPass.fwdPassPreOutputAsArrays[time];
				deltaNext.assign(afn.backprop(oz.dup('f'), dLdoNext.dup('f')).getFirst());
			}

			if (maskArray != null) {
				timeStepMaskColumn = maskArray.getColumn(time);
				deltaNext.muliColumnVector(timeStepMaskColumn);
			}

			INDArray prevLayerActivationSlice = Shape
					.toMmulCompatible(is2dInput ? input : input.tensorAlongDimension(time, 1, 0));

			Nd4j.gemm(prevLayerActivationSlice, deltaNext, iwGradientsOut, true, false, 1.0, 1.0);

			INDArray prevAct = iTimeIndex == 0 ? fwdPass.prevAct : fwdPass.fwdPassOutputAsArrays[time - inext];
			if (iTimeIndex > 0 || prevAct != null) {
				Nd4j.gemm(prevAct, deltaNext, rwGradientsOut, true, false, 1.0, 1.0);
			}

			l1BLAS.axpy(hiddenLayerSize, 1.0, deltaNext.sum(0), bGradientsOut);

			INDArray epsilonNextSlice = epsilonNext.tensorAlongDimension(time, 1, 0);
			Nd4j.gemm(deltaNext, inputWeights, epsilonNextSlice, false, true, 1.0, 1.0);

			if (maskArray != null) {
				epsilonNextSlice.muliColumnVector(timeStepMaskColumn);
			}
		}

		Gradient retGradient = new DefaultGradient();
		retGradient.gradientForVariable().put(inputWeightKey, iwGradientsOut);
		retGradient.gradientForVariable().put(recurrentWeightKey, rwGradientsOut);
		retGradient.gradientForVariable().put(biasWeightKey, bGradientsOut);

		return new Pair<>(retGradient, epsilonNext);
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

	protected RNNFwdPassReturn activateHelper(final boolean training, final INDArray prevOutputActivations,
			boolean forBackprop) {
		if (cacheMode == null)
			cacheMode = CacheMode.NONE;

		if (forBackprop && cachedFwdPass != null) {
			RNNFwdPassReturn ret = cachedFwdPass;
			cachedFwdPass = null;
			return ret;
		}

		final INDArray recurrentWeights = getParam(RNNParamInitializer.RECURRENT_WEIGHT_KEY);
		final INDArray inputWeights = getParam(RNNParamInitializer.INPUT_WEIGHT_KEY);
		final INDArray biases = getParam(RNNParamInitializer.BIAS_KEY);

		RNNFwdPassReturn fwd = activateHelper(this, this.conf, this.input, recurrentWeights, inputWeights, biases,
				training, prevOutputActivations, forBackprop || (cacheMode != CacheMode.NONE && training),
				RNNParamInitializer.INPUT_WEIGHT_KEY, maskArray, forBackprop ? cacheMode : CacheMode.NONE);

		if (training && cacheMode != CacheMode.NONE) {
			cachedFwdPass = fwd;
		}

		return fwd;
	}

	private static RNNFwdPassReturn activateHelper(final RNN layer, final NeuralNetConfiguration conf,
			final INDArray input, final INDArray recurrentWeights, final INDArray originalInputWeights,
			final INDArray biases, final boolean training, final INDArray originalPrevOutputActivations,
			boolean forBackprop, final String inputWeightKey, INDArray maskArray, final CacheMode cacheMode) {

		if (input == null || input.length() == 0) {
			throw new IllegalArgumentException("Invalid input: not set or 0 length");
		}

		INDArray inputWeights = originalInputWeights;
		INDArray prevOutputActivations = originalPrevOutputActivations;

		if (input.size(1) != inputWeights.size(0)) {
			throw new DL4JInvalidInputException("Received input with size(1) = " + input.size(1)
					+ " (input array shape = " + Arrays.toString(input.shape())
					+ "); input.size(1) must match layer nIn size (nIn = " + inputWeights.size(0) + ")");
		}

		if (prevOutputActivations != null && prevOutputActivations.size(0) != input.size(0)) {
			throw new DL4JInvalidInputException("Previous activations (stored state) number of examples = "
					+ prevOutputActivations.size(0) + " but input array number of examples = " + input.size(0)
					+ ". Possible cause: using rnnTimeStep() without calling"
					+ " rnnClearPreviousState() between different sequences?");
		}

		RNNFwdPassReturn toReturn = new RNNFwdPassReturn();

		boolean is2dInput = input.rank() < 3;
		int timeSeriesLength = (is2dInput ? 1 : input.size(2));
		int hiddenLayerSize = recurrentWeights.size(0);
		int miniBatchSize = input.size(0);

		INDArray outputActivations = null;
		IActivation afn = layer.layerConf().getActivationFn();

		if (forBackprop) {
			toReturn.fwdPassPreOutputAsArrays = new INDArray[timeSeriesLength];
			toReturn.fwdPassOutputAsArrays = new INDArray[timeSeriesLength];

			if (cacheMode != CacheMode.NONE) {
				try (MemoryWorkspace ws = Nd4j.getWorkspaceManager()
						.getWorkspaceForCurrentThread(ComputationGraph.WORKSPACE_CACHE).notifyScopeBorrowed()) {
					outputActivations = Nd4j.create(new int[] { miniBatchSize, hiddenLayerSize, timeSeriesLength },
							'f');
					toReturn.fwdPassOutput = outputActivations;
				}
			}
		} else {
			outputActivations = Nd4j.create(new int[] { miniBatchSize, hiddenLayerSize, timeSeriesLength }, 'f');
			toReturn.fwdPassOutput = outputActivations;
		}

		if (prevOutputActivations == null) {
			prevOutputActivations = Nd4j.zeros(new int[] { miniBatchSize, hiddenLayerSize });
		}

		for (int iTimeIndex = 0; iTimeIndex < timeSeriesLength; iTimeIndex++) {
			int time = iTimeIndex;

			INDArray miniBatchData = (is2dInput ? input : input.tensorAlongDimension(time, 1, 0));
			miniBatchData = Shape.toMmulCompatible(miniBatchData);

			INDArray oz = miniBatchData.mmul(inputWeights);
			Nd4j.gemm(prevOutputActivations, recurrentWeights, oz, false, false, 1.0, 1.0);
			oz.addiRowVector(biases);
			INDArray currentOutputActivations = afn.getActivation(oz.dup('f'), training);

			if (forBackprop) {
				toReturn.fwdPassPreOutputAsArrays[time] = oz;
				toReturn.fwdPassOutputAsArrays[time] = currentOutputActivations;

				if (cacheMode != CacheMode.NONE) {
					outputActivations.tensorAlongDimension(time, 1, 0).assign(currentOutputActivations);
				}
			} else {
				outputActivations.tensorAlongDimension(time, 1, 0).assign(currentOutputActivations);
			}

			prevOutputActivations = currentOutputActivations;

			toReturn.lastAct = currentOutputActivations;
		}

		toReturn.prevAct = originalPrevOutputActivations;

		return toReturn;
	}

	@Override
	public Pair<INDArray, MaskState> feedForwardMaskArray(INDArray maskArray, MaskState currentMaskState,
			int minibatchSize) {
		return new Pair<>(maskArray, MaskState.Passthrough);
	}
}