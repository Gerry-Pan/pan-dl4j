package personal.pan.dl4j.nn.layers.recurrent;

import java.util.Arrays;
import java.util.Map;

import org.deeplearning4j.exception.DL4JInvalidInputException;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.blas.Level1;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.same.TimesOneMinus;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

/**
 * created on 2017-12-29
 * 
 * @author Gerry Pan
 *
 */
public class GRUHelpers {

	public static GRUFwdPassReturn activateHelper(final GRU layer, final NeuralNetConfiguration conf,
			final IActivation gateActivationFn, final INDArray input, final INDArray recurrentWeights,
			final INDArray originalInputWeights, final INDArray biases, final boolean training,
			final INDArray originalPrevOutputActivations, boolean forBackprop, final String inputWeightKey,
			INDArray maskArray, final CacheMode cacheMode, final LayerWorkspaceMgr workspaceMgr, DataType dataType) {

		if (input == null || input.length() == 0) {
			throw new IllegalArgumentException("Invalid input: not set or 0 length");
		}

		GRUFwdPassReturn toReturn = new GRUFwdPassReturn();

		INDArray inputWeights = originalInputWeights;
		INDArray prevOutputActivations = originalPrevOutputActivations;

		boolean is2dInput = input.rank() < 3;
		int timeSeriesLength = (int) (is2dInput ? 1 : input.size(2));
		int hiddenLayerSize = (int) recurrentWeights.size(0);
		int miniBatchSize = (int) input.size(0);

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

		INDArray outputActivations = null;
		IActivation afn = layer.layerConf().getActivationFn();

		if (forBackprop) {
			toReturn.fwdPassOutputAsArrays = new INDArray[timeSeriesLength];
			toReturn.rz = new INDArray[timeSeriesLength];
			toReturn.ra = new INDArray[timeSeriesLength];
			toReturn.uz = new INDArray[timeSeriesLength];
			toReturn.ua = new INDArray[timeSeriesLength];
			toReturn.hz = new INDArray[timeSeriesLength];
			toReturn.ha = new INDArray[timeSeriesLength];

			if (training && cacheMode != CacheMode.NONE && workspaceMgr.hasConfiguration(ArrayType.FF_CACHE)
					&& workspaceMgr.isWorkspaceOpen(ArrayType.FF_CACHE)) {
				try (MemoryWorkspace wsB = workspaceMgr.notifyScopeBorrowed(ArrayType.FF_CACHE)) {
					outputActivations = Nd4j.create(new long[] { miniBatchSize, hiddenLayerSize, timeSeriesLength },
							'f');
					toReturn.fwdPassOutput = outputActivations;
				}
			} else {
				outputActivations = workspaceMgr.create(ArrayType.ACTIVATIONS, dataType,
						new long[] { miniBatchSize, hiddenLayerSize, timeSeriesLength }, 'f'); // F order to keep time
																								// steps together
				toReturn.fwdPassOutput = outputActivations;
			}
		} else {
			outputActivations = workspaceMgr.create(ArrayType.ACTIVATIONS, dataType,
					new long[] { miniBatchSize, hiddenLayerSize, timeSeriesLength }, 'f');
			toReturn.fwdPassOutput = outputActivations;
		}

		INDArray inputWeightsR = inputWeights.get(NDArrayIndex.all(), NDArrayIndex.interval(0, hiddenLayerSize));
		INDArray inputWeightsU = inputWeights.get(NDArrayIndex.all(),
				NDArrayIndex.interval(hiddenLayerSize, 2 * hiddenLayerSize));
		INDArray inputWeightsH = inputWeights.get(NDArrayIndex.all(),
				NDArrayIndex.interval(2 * hiddenLayerSize, 3 * hiddenLayerSize));

		INDArray recurrentWeightsR = recurrentWeights.get(NDArrayIndex.all(),
				NDArrayIndex.interval(0, hiddenLayerSize));
		INDArray recurrentWeightsU = recurrentWeights.get(NDArrayIndex.all(),
				NDArrayIndex.interval(hiddenLayerSize, 2 * hiddenLayerSize));
		INDArray recurrentWeightsH = recurrentWeights.get(NDArrayIndex.all(),
				NDArrayIndex.interval(2 * hiddenLayerSize, 3 * hiddenLayerSize));

		INDArray biasesR = biases.get(NDArrayIndex.all(), NDArrayIndex.interval(0, hiddenLayerSize));
		INDArray biasesU = biases.get(NDArrayIndex.all(), NDArrayIndex.interval(hiddenLayerSize, 2 * hiddenLayerSize));
		INDArray biasesH = biases.get(NDArrayIndex.all(),
				NDArrayIndex.interval(2 * hiddenLayerSize, 3 * hiddenLayerSize));

		if (prevOutputActivations == null) {
			prevOutputActivations = Nd4j.zeros(new long[] { miniBatchSize, hiddenLayerSize });
		}

		for (int iTimeIndex = 0; iTimeIndex < timeSeriesLength; iTimeIndex++) {
			int time = iTimeIndex;

			INDArray miniBatchData = (is2dInput ? input : input.tensorAlongDimension(time, 1, 0));
			miniBatchData = Shape.toMmulCompatible(miniBatchData);

			// reset gate
			INDArray rz = miniBatchData.mmul(inputWeightsR);
			Nd4j.gemm(prevOutputActivations, recurrentWeightsR, rz, false, false, 1.0, 1.0);
			rz.addiRowVector(biasesR);
			INDArray ra = gateActivationFn.getActivation(rz.dup('f'), training);

			// update gate
			INDArray uz = miniBatchData.mmul(inputWeightsU);
			Nd4j.gemm(prevOutputActivations, recurrentWeightsU, uz, false, false, 1.0, 1.0);
			uz.addiRowVector(biasesU);
			INDArray ua = gateActivationFn.getActivation(uz.dup('f'), training);

			// candidate activation
			INDArray hz = miniBatchData.mmul(inputWeightsH);
			Nd4j.gemm(prevOutputActivations.mul(ra), recurrentWeightsH, hz, false, false, 1.0, 1.0);
			hz.addiRowVector(biasesH);
			INDArray ha = afn.getActivation(hz.dup('f'), training);

			if (forBackprop) {
				toReturn.rz[time] = rz;
				toReturn.ra[time] = ra;

				toReturn.uz[time] = uz;
				toReturn.ua[time] = ua;

				toReturn.hz[time] = hz;
				toReturn.ha[time] = ha;
			}

			INDArray currentOutputActivations = ua.mul(-1).add(1);
			currentOutputActivations.muli(prevOutputActivations);
			currentOutputActivations.addi(ua.mul(ha));

			if (maskArray != null) {
				INDArray timeStepMaskColumn = maskArray.getColumn(time);
				currentOutputActivations.muliColumnVector(timeStepMaskColumn);
			}

			if (forBackprop) {
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

	/**
	 * 
	 * @param conf
	 * @param gateActivationFn
	 * @param input               shape(batchSize,nLast,timeSeriesLength)
	 * @param recurrentWeights    shape(hiddenLayerSize,3*hiddenLayerSize)
	 * @param inputWeights        shape(nLast,3*hiddenLayerSize)
	 * @param epsilon             是损失函数对后一层输入值的偏导，shape(batchSize,hiddenLayerSize,timeSeriesLength)
	 * @param truncatedBPTT
	 * @param tbpttBackwardLength
	 * @param fwdPass
	 * @param inputWeightKey
	 * @param recurrentWeightKey
	 * @param biasWeightKey
	 * @param gradientViews
	 * @param maskArray
	 * @return
	 */
	public static Pair<Gradient, INDArray> backpropGradientHelper(final NeuralNetConfiguration conf,
			final IActivation gateActivationFn, final INDArray input, final INDArray recurrentWeights,
			final INDArray inputWeights, final INDArray epsilon, final boolean truncatedBPTT,
			final int tbpttBackwardLength, final GRUFwdPassReturn fwdPass, final String inputWeightKey,
			final String recurrentWeightKey, final String biasWeightKey, final Map<String, INDArray> gradientViews,
			INDArray maskArray, final LayerWorkspaceMgr workspaceMgr, DataType dataType) {

		long miniBatchSize = epsilon.size(0);
		boolean is2dInput = epsilon.rank() < 3;
		long prevLayerSize = inputWeights.size(0);// prevLayerSize=nLast
		long hiddenLayerSize = recurrentWeights.size(0);
		long timeSeriesLength = (is2dInput ? 1 : epsilon.size(2));

		Level1 l1BLAS = Nd4j.getBlasWrapper().level1();

		INDArray iwGradientsOut = gradientViews.get(inputWeightKey);// shape(nLast,3*hiddenLayerSize)
		INDArray rwGradientsOut = gradientViews.get(recurrentWeightKey);// shape(hiddenLayerSize,3*hiddenLayerSize)
		INDArray bGradientsOut = gradientViews.get(biasWeightKey);
		iwGradientsOut.assign(0);
		rwGradientsOut.assign(0);
		bGradientsOut.assign(0);

		/*
		 * recurrentWeightsR|recurrentWeightsU|recurrentWeightsH
		 * shape(hiddenLayerSize,hiddenLayerSize)
		 */
		INDArray recurrentWeightsR = recurrentWeights.get(NDArrayIndex.all(),
				NDArrayIndex.interval(0, hiddenLayerSize));
		INDArray recurrentWeightsU = recurrentWeights.get(NDArrayIndex.all(),
				NDArrayIndex.interval(hiddenLayerSize, 2 * hiddenLayerSize));
		INDArray recurrentWeightsH = recurrentWeights.get(NDArrayIndex.all(),
				NDArrayIndex.interval(2 * hiddenLayerSize, 3 * hiddenLayerSize));

		/*
		 * deltarNext|deltauNext|deltahNext,shape(batchSize,hiddenLayerSize)
		 */
		INDArray deltaruhNext = Nd4j.create(new long[] { miniBatchSize, 3 * hiddenLayerSize }, 'f');
		INDArray deltarNext = deltaruhNext.get(NDArrayIndex.all(), NDArrayIndex.interval(0, hiddenLayerSize));
		INDArray deltauNext = deltaruhNext.get(NDArrayIndex.all(),
				NDArrayIndex.interval(hiddenLayerSize, 2 * hiddenLayerSize));
		INDArray deltahNext = deltaruhNext.get(NDArrayIndex.all(),
				NDArrayIndex.interval(2 * hiddenLayerSize, 3 * hiddenLayerSize));

		long endIdx = 0;

		if (truncatedBPTT) {
			endIdx = Math.max(0, timeSeriesLength - tbpttBackwardLength);
		}

		INDArray dLdoNext = null;

		INDArray timeStepMaskColumn = null;
		INDArray epsilonNext = workspaceMgr.create(ArrayType.ACTIVATION_GRAD, dataType,
				new long[] { miniBatchSize, prevLayerSize, timeSeriesLength }, 'f');
		IActivation afn = ((org.deeplearning4j.nn.conf.layers.BaseLayer) conf.getLayer()).getActivationFn();

		for (long iTimeIndex = timeSeriesLength - 1; iTimeIndex >= endIdx; iTimeIndex--) {
			try (MemoryWorkspace ws = workspaceMgr.notifyScopeEntered(ArrayType.RNN_BP_LOOP_WORKING_MEM)) {
				int time = (int) iTimeIndex;
				int inext = 1;

				/*
				 * dLdo是损失函数对本层当前时刻输出值的偏导，故dLdo也是损失函数对本层后一时刻输入值的偏导
				 * dLdo与GRUFwdPassReturn.fwdPassOutput有相同的shape，shape(batchSize,hiddenLayerSize)
				 * epsilon是损失函数对后一层输入值的偏导，shape(batchSize,hiddenLayerSize,timeSeriesLength)
				 * epsilonSlice为epsilon的时间切片，shape(batchSize,hiddenLayerSize)
				 */
				INDArray dLdo = null;
				INDArray epsilonSlice = (is2dInput ? epsilon : epsilon.tensorAlongDimension(time, 1, 0));

				if (iTimeIndex != timeSeriesLength - 1) {
					INDArray uaNext = fwdPass.ua[time + 1];
					dLdo = uaNext.mul(-1);
					dLdo.addi(1);
					dLdo.muli(dLdoNext);
				} else {
					dLdo = Nd4j.create(new long[] { miniBatchSize, hiddenLayerSize }, 'f');
				}

				dLdo.addi(Shape.toOffsetZeroCopy(epsilonSlice, 'f'));

				INDArray raNext = null;// shape(batchSize,hiddenLayerSize)

				if (iTimeIndex != timeSeriesLength - 1) {
					raNext = fwdPass.ra[time + 1];
				} else {
					raNext = Nd4j.create(new long[] { miniBatchSize, hiddenLayerSize }, 'f');
				}

				INDArray temp = Nd4j.create(new long[] { miniBatchSize, hiddenLayerSize }, 'f');

				/*
				 * (batchSize,hiddenLayerSize) * (hiddenLayerSize,hiddenLayerSize)^T =
				 * (batchSize,hiddenLayerSize)
				 */
				Nd4j.gemm(deltarNext, recurrentWeightsR, dLdo, false, true, 1.0, 1.0);
				Nd4j.gemm(deltauNext, recurrentWeightsU, dLdo, false, true, 1.0, 1.0);
				Nd4j.gemm(deltahNext, recurrentWeightsH, temp, false, true, 1.0, 1.0);

				temp.muli(raNext);
				dLdo.addi(temp);
				dLdoNext = workspaceMgr.leverageTo(ArrayType.BP_WORKING_MEM, dLdo);

				INDArray ha = null;
				if (iTimeIndex != timeSeriesLength - 1) {
					ha = fwdPass.ha[time];
				} else {
					ha = Nd4j.create(new long[] { miniBatchSize, hiddenLayerSize }, 'f');
				}

				INDArray uz = fwdPass.uz[time];
				INDArray ua = fwdPass.ua[time];
				INDArray prevAct = iTimeIndex == 0 ? fwdPass.prevAct : fwdPass.fwdPassOutputAsArrays[time - inext];// shape(batchSize,hiddenLayerSize)

				INDArray deltau = dLdo.dup('f');
				if (iTimeIndex > 0 || prevAct != null) {
					deltau.muli(ha.sub(prevAct));
				} else {
					deltau.muli(ha);
				}
				deltau.assign(gateActivationFn.backprop(uz.dup('f'), deltau).getFirst());

				INDArray hz = fwdPass.hz[time];
				INDArray deltah = dLdo.mul(ua);
				deltah.assign(afn.backprop(hz.dup('f'), deltah).getFirst());

				INDArray rz = fwdPass.rz[time];
				INDArray ra = fwdPass.ra[time];
				temp = Nd4j.getExecutioner().exec(new TimesOneMinus(ra.dup('f')));
				INDArray deltar = Nd4j.create(new long[] { miniBatchSize, hiddenLayerSize }, 'f');

				if (prevAct != null) {
					Nd4j.gemm(deltah, recurrentWeightsH, deltar, false, true, 1.0, 1.0);
					deltar.muli(prevAct);
					deltar.assign(gateActivationFn.backprop(rz.dup('f'), deltar).getFirst());
				}

				deltarNext.assign(deltar);
				deltauNext.assign(deltau);
				deltahNext.assign(deltah);

				if (maskArray != null) {
					timeStepMaskColumn = maskArray.getColumn(time);
					deltaruhNext.muliColumnVector(timeStepMaskColumn);
				}

				INDArray prevLayerActivationSlice = Shape
						.toMmulCompatible(is2dInput ? input : input.tensorAlongDimension(time, 1, 0));

				if (iTimeIndex > 0 || prevAct != null) {
					Nd4j.gemm(prevLayerActivationSlice, deltaruhNext, iwGradientsOut, true, false, 1.0, 1.0);
				} else {
					INDArray iwGradientsOutU = iwGradientsOut.get(NDArrayIndex.all(),
							NDArrayIndex.interval(hiddenLayerSize, 2 * hiddenLayerSize));
					INDArray iwGradientsOutH = iwGradientsOut.get(NDArrayIndex.all(),
							NDArrayIndex.interval(2 * hiddenLayerSize, 3 * hiddenLayerSize));

					Nd4j.gemm(prevLayerActivationSlice, deltauNext, iwGradientsOutU, true, false, 1.0, 1.0);
					Nd4j.gemm(prevLayerActivationSlice, deltahNext, iwGradientsOutH, true, false, 1.0, 1.0);
				}

				if (iTimeIndex > 0 || prevAct != null) {

					/*
					 * rwGradientsOutR|rwGradientsOutU|rwGradientsOutH
					 * shape(hiddenLayerSize,hiddenLayerSize)
					 */
					INDArray rwGradientsOutR = rwGradientsOut.get(NDArrayIndex.all(),
							NDArrayIndex.interval(0, hiddenLayerSize));
					INDArray rwGradientsOutU = rwGradientsOut.get(NDArrayIndex.all(),
							NDArrayIndex.interval(hiddenLayerSize, 2 * hiddenLayerSize));
					INDArray rwGradientsOutH = rwGradientsOut.get(NDArrayIndex.all(),
							NDArrayIndex.interval(2 * hiddenLayerSize, 3 * hiddenLayerSize));

					/*
					 * (batchSize,hiddenLayerSize)^T * (batchSize,hiddenLayerSize) =
					 * (hiddenLayerSize,hiddenLayerSize)
					 */
					Nd4j.gemm(prevAct, deltarNext, rwGradientsOutR, true, false, 1.0, 1.0);
					Nd4j.gemm(prevAct, deltauNext, rwGradientsOutU, true, false, 1.0, 1.0);
					Nd4j.gemm(prevAct.mul(ra), deltahNext, rwGradientsOutH, true, false, 1.0, 1.0);
				}

				if (iTimeIndex > 0 || prevAct != null) {
					l1BLAS.axpy(3 * hiddenLayerSize, 1.0, deltaruhNext.sum(0), bGradientsOut);
				} else {
					INDArray uBiasGrad = bGradientsOut.get(NDArrayIndex.point(0),
							NDArrayIndex.interval(hiddenLayerSize, 2 * hiddenLayerSize));

					INDArray hBiasGrad = bGradientsOut.get(NDArrayIndex.point(0),
							NDArrayIndex.interval(2 * hiddenLayerSize, 3 * hiddenLayerSize));

					l1BLAS.axpy(hiddenLayerSize, 1.0, deltau.sum(0), uBiasGrad);
					l1BLAS.axpy(hiddenLayerSize, 1.0, deltah.sum(0), hBiasGrad);
				}

				/*
				 * epsilonNextSlice = shape(batchSize, nLast)
				 */
				INDArray epsilonNextSlice = epsilonNext.tensorAlongDimension(time, 1, 0);

				if (iTimeIndex > 0 || prevAct != null) {
					Nd4j.gemm(deltaruhNext, inputWeights, epsilonNextSlice, false, true, 1.0, 1.0);
				} else {
					INDArray inputWeightsUH = inputWeights.get(NDArrayIndex.all(),
							NDArrayIndex.interval(hiddenLayerSize, 3 * hiddenLayerSize));

					INDArray deltauhNext = deltaruhNext.get(NDArrayIndex.all(),
							NDArrayIndex.interval(hiddenLayerSize, 3 * hiddenLayerSize));

					Nd4j.gemm(deltauhNext, inputWeightsUH, epsilonNextSlice, false, true, 1.0, 1.0);
				}

				if (maskArray != null) {
					epsilonNextSlice.muliColumnVector(timeStepMaskColumn);
				}
			}
		}

		Gradient retGradient = new DefaultGradient();
		retGradient.gradientForVariable().put(inputWeightKey, iwGradientsOut);
		retGradient.gradientForVariable().put(recurrentWeightKey, rwGradientsOut);
		retGradient.gradientForVariable().put(biasWeightKey, bGradientsOut);

		return new Pair<>(retGradient, epsilonNext);
	}

	public static LayerMemoryReport getMemoryReport(org.deeplearning4j.nn.conf.layers.FeedForwardLayer lstmLayer,
			InputType inputType) {
		// do it in future
		return null;
	}
}
