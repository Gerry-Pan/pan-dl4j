package personal.pan.dl4j.nn.params;

import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.IWeightInit;
import org.deeplearning4j.nn.weights.WeightInitUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import personal.pan.dl4j.nn.conf.layers.GRU;

/**
 * created on 2017-12-29
 * 
 * @author Gerry Pan
 *
 */
public class GRUParamInitializer implements ParamInitializer {

	public static final GRUParamInitializer INSTANCE = new GRUParamInitializer();

	public final static String RECURRENT_WEIGHT_KEY = "RW";
	public final static String BIAS_KEY = DefaultParamInitializer.BIAS_KEY;
	public final static String INPUT_WEIGHT_KEY = DefaultParamInitializer.WEIGHT_KEY;

	public static GRUParamInitializer getInstance() {
		return INSTANCE;
	}

	@Override
	public long numParams(NeuralNetConfiguration conf) {
		return numParams(conf.getLayer());
	}

	@Override
	public long numParams(Layer l) {
		GRU layer = (GRU) l;

		long nL = layer.getNOut();
		long nLast = layer.getNIn();

		long nParams = nLast * (3 * nL) + nL * (3 * nL) + 3 * nL;
		return nParams;
	}

	@Override
	public Map<String, INDArray> init(NeuralNetConfiguration conf, INDArray paramsView, boolean initializeParams) {
		GRU layer = (GRU) conf.getLayer();
		Map<String, INDArray> params = Collections.synchronizedMap(new LinkedHashMap<String, INDArray>());

		long nL = layer.getNOut();
		long nLast = layer.getNIn();

		conf.addVariable(INPUT_WEIGHT_KEY);
		conf.addVariable(RECURRENT_WEIGHT_KEY);
		conf.addVariable(BIAS_KEY);

		long length = numParams(conf);
		if (paramsView.length() != length)
			throw new IllegalStateException(
					"Expected params view of length " + length + ", got length " + paramsView.length());

		long nParamsIn = nLast * (3 * nL);
		long nParamsRecurrent = nL * (3 * nL);
		long nBias = 3 * nL;
		INDArray inputWeightView = paramsView.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, nParamsIn));
		INDArray recurrentWeightView = paramsView.get(NDArrayIndex.point(0),
				NDArrayIndex.interval(nParamsIn, nParamsIn + nParamsRecurrent));
		INDArray biasView = paramsView.get(NDArrayIndex.point(0),
				NDArrayIndex.interval(nParamsIn + nParamsRecurrent, nParamsIn + nParamsRecurrent + nBias));

		if (initializeParams) {
			long fanIn = nL;
			long fanOut = nLast + nL;
			long[] inputWShape = new long[] { nLast, 3 * nL };
			long[] recurrentWShape = new long[] { nL, 3 * nL };

			IWeightInit rwInit;
			if (layer.getWeightInitFnRecurrent() != null) {
				rwInit = layer.getWeightInitFnRecurrent();
			} else {
				rwInit = layer.getWeightInitFn();
			}

			params.put(INPUT_WEIGHT_KEY, layer.getWeightInitFn().init(fanIn, fanOut, inputWShape,
					IWeightInit.DEFAULT_WEIGHT_INIT_ORDER, inputWeightView));
			params.put(RECURRENT_WEIGHT_KEY, rwInit.init(fanIn, fanOut, recurrentWShape,
					IWeightInit.DEFAULT_WEIGHT_INIT_ORDER, recurrentWeightView));
			biasView.put(new INDArrayIndex[] { NDArrayIndex.point(0), NDArrayIndex.interval(nL, 2 * nL) },
					Nd4j.valueArrayOf(1, nL, 1));

			params.put(BIAS_KEY, biasView);
		} else {
			params.put(INPUT_WEIGHT_KEY, WeightInitUtil.reshapeWeights(new long[] { nLast, 3 * nL }, inputWeightView));
			params.put(RECURRENT_WEIGHT_KEY,
					WeightInitUtil.reshapeWeights(new long[] { nL, 3 * nL }, recurrentWeightView));
			params.put(BIAS_KEY, biasView);
		}

		return params;
	}

	@Override
	public Map<String, INDArray> getGradientsFromFlattened(NeuralNetConfiguration conf, INDArray gradientView) {
		GRU layer = (GRU) conf.getLayer();

		long nL = layer.getNOut();
		long nLast = layer.getNIn();

		long length = numParams(conf);
		if (gradientView.length() != length)
			throw new IllegalStateException(
					"Expected gradient view of length " + length + ", got length " + gradientView.length());

		long nParamsIn = nLast * (3 * nL);
		long nParamsRecurrent = nL * (3 * nL);
		long nBias = 3 * nL;
		INDArray inputWeightGradView = gradientView.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, nParamsIn))
				.reshape('f', nLast, 3 * nL);
		INDArray recurrentWeightGradView = gradientView
				.get(NDArrayIndex.point(0), NDArrayIndex.interval(nParamsIn, nParamsIn + nParamsRecurrent))
				.reshape('f', nL, 3 * nL);
		INDArray biasGradView = gradientView.get(NDArrayIndex.point(0),
				NDArrayIndex.interval(nParamsIn + nParamsRecurrent, nParamsIn + nParamsRecurrent + nBias));

		Map<String, INDArray> out = new LinkedHashMap<>();
		out.put(INPUT_WEIGHT_KEY, inputWeightGradView);
		out.put(RECURRENT_WEIGHT_KEY, recurrentWeightGradView);
		out.put(BIAS_KEY, biasGradView);

		return out;
	}

	@Override
	public List<String> paramKeys(Layer layer) {
		return Arrays.asList(INPUT_WEIGHT_KEY, RECURRENT_WEIGHT_KEY, BIAS_KEY);
	}

	@Override
	public List<String> weightKeys(Layer layer) {
		return Arrays.asList(INPUT_WEIGHT_KEY, RECURRENT_WEIGHT_KEY);
	}

	@Override
	public List<String> biasKeys(Layer layer) {
		return Collections.singletonList(BIAS_KEY);
	}

	@Override
	public boolean isWeightParam(Layer layer, String key) {
		return RECURRENT_WEIGHT_KEY.equals(key) || INPUT_WEIGHT_KEY.equals(key);
	}

	@Override
	public boolean isBiasParam(Layer layer, String key) {
		return BIAS_KEY.equals(key);
	}

}
