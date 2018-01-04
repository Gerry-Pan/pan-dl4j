package personal.pan.dl4j.nn.params;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.Distributions;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInitUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import personal.pan.dl4j.nn.conf.layers.RNN;

/**
 * created on 2018-1-3
 * 
 * @author Gerry Pan
 *
 */
public class RNNParamInitializer implements ParamInitializer {

	public static final RNNParamInitializer INSTANCE = new RNNParamInitializer();

	public final static String RECURRENT_WEIGHT_KEY = "RW";
	public final static String BIAS_KEY = DefaultParamInitializer.BIAS_KEY;
	public final static String INPUT_WEIGHT_KEY = DefaultParamInitializer.WEIGHT_KEY;

	public static RNNParamInitializer getInstance() {
		return INSTANCE;
	}

	@Override
	public int numParams(NeuralNetConfiguration conf) {
		return numParams(conf.getLayer());
	}

	@Override
	public int numParams(Layer l) {
		RNN layer = (RNN) l;

		int nL = layer.getNOut();
		int nLast = layer.getNIn();

		int nParams = nLast * nL + nL * nL + nL;
		return nParams;
	}

	@Override
	public Map<String, INDArray> init(NeuralNetConfiguration conf, INDArray paramsView, boolean initializeParams) {
		RNN layer = (RNN) conf.getLayer();

		Map<String, INDArray> params = Collections.synchronizedMap(new LinkedHashMap<String, INDArray>());

		Distribution dist = Distributions.createDistribution(layer.getDist());

		int nL = layer.getNOut();
		int nLast = layer.getNIn();

		conf.addVariable(INPUT_WEIGHT_KEY);
		conf.addVariable(RECURRENT_WEIGHT_KEY);
		conf.addVariable(BIAS_KEY);

		int length = numParams(conf);
		if (paramsView.length() != length)
			throw new IllegalStateException(
					"Expected params view of length " + length + ", got length " + paramsView.length());

		int nParamsIn = nLast * nL;
		int nParamsRecurrent = nL * nL;
		int nBias = nL;
		INDArray inputWeightView = paramsView.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, nParamsIn));
		INDArray recurrentWeightView = paramsView.get(NDArrayIndex.point(0),
				NDArrayIndex.interval(nParamsIn, nParamsIn + nParamsRecurrent));
		INDArray biasView = paramsView.get(NDArrayIndex.point(0),
				NDArrayIndex.interval(nParamsIn + nParamsRecurrent, nParamsIn + nParamsRecurrent + nBias));

		if (initializeParams) {
			int fanIn = nL;
			int fanOut = nLast + nL;
			int[] inputWShape = new int[] { nLast, nL };
			int[] recurrentWShape = new int[] { nL, nL };

			params.put(INPUT_WEIGHT_KEY, WeightInitUtil.initWeights(fanIn, fanOut, inputWShape, layer.getWeightInit(),
					dist, inputWeightView));
			params.put(RECURRENT_WEIGHT_KEY, WeightInitUtil.initWeights(fanIn, fanOut, recurrentWShape,
					layer.getWeightInit(), dist, recurrentWeightView));
			biasView.put(new INDArrayIndex[] { NDArrayIndex.point(0), NDArrayIndex.interval(0, nL) },
					Nd4j.valueArrayOf(1, nL, 1));

			params.put(BIAS_KEY, biasView);
		} else {
			params.put(INPUT_WEIGHT_KEY, WeightInitUtil.reshapeWeights(new int[] { nLast, nL }, inputWeightView));
			params.put(RECURRENT_WEIGHT_KEY, WeightInitUtil.reshapeWeights(new int[] { nL, nL }, recurrentWeightView));
			params.put(BIAS_KEY, biasView);
		}

		return params;
	}

	@Override
	public Map<String, INDArray> getGradientsFromFlattened(NeuralNetConfiguration conf, INDArray gradientView) {
		RNN layer = (RNN) conf.getLayer();

		int nL = layer.getNOut();
		int nLast = layer.getNIn();

		int length = numParams(conf);
		if (gradientView.length() != length)
			throw new IllegalStateException(
					"Expected gradient view of length " + length + ", got length " + gradientView.length());

		int nParamsIn = nLast * nL;
		int nParamsRecurrent = nL * nL;
		int nBias = nL;
		INDArray inputWeightGradView = gradientView.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, nParamsIn))
				.reshape('f', nLast, nL);
		INDArray recurrentWeightGradView = gradientView
				.get(NDArrayIndex.point(0), NDArrayIndex.interval(nParamsIn, nParamsIn + nParamsRecurrent))
				.reshape('f', nL, nL);
		INDArray biasGradView = gradientView.get(NDArrayIndex.point(0),
				NDArrayIndex.interval(nParamsIn + nParamsRecurrent, nParamsIn + nParamsRecurrent + nBias));

		Map<String, INDArray> out = new LinkedHashMap<>();
		out.put(INPUT_WEIGHT_KEY, inputWeightGradView);
		out.put(RECURRENT_WEIGHT_KEY, recurrentWeightGradView);
		out.put(BIAS_KEY, biasGradView);

		return out;
	}

}
