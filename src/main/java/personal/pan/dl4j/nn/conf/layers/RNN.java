package personal.pan.dl4j.nn.conf.layers;

import java.util.Collection;
import java.util.Map;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.BaseRecurrentLayer;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;

import personal.pan.dl4j.nn.params.RNNParamInitializer;

/**
 * 原始的RNN算法<br>
 * created on 2018-1-3
 * 
 * @author Gerry Pan
 *
 */
public class RNN extends BaseRecurrentLayer {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public RNN() {
		super();
	}

	public RNN(Builder builder) {
		super(builder);
	}

	@Override
	public Layer instantiate(NeuralNetConfiguration conf, Collection<IterationListener> iterationListeners,
			int layerIndex, INDArray layerParamsView, boolean initializeParams) {
		personal.pan.dl4j.nn.layers.recurrent.RNN layer = new personal.pan.dl4j.nn.layers.recurrent.RNN(conf);
		layer.setListeners(iterationListeners);
		layer.setIndex(layerIndex);
		layer.setParamsViewArray(layerParamsView);
		Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
		layer.setParamTable(paramTable);
		layer.setConf(conf);
		return layer;
	}

	@Override
	public ParamInitializer initializer() {
		return RNNParamInitializer.INSTANCE;
	}

	@Override
	public double getL1ByParam(String paramName) {
		switch (paramName) {
		case RNNParamInitializer.INPUT_WEIGHT_KEY:
		case RNNParamInitializer.RECURRENT_WEIGHT_KEY:
			return l1;
		case RNNParamInitializer.BIAS_KEY:
			return l1Bias;
		default:
			throw new IllegalArgumentException("Unknown parameter name: \"" + paramName + "\"");
		}
	}

	@Override
	public double getL2ByParam(String paramName) {
		switch (paramName) {
		case RNNParamInitializer.INPUT_WEIGHT_KEY:
		case RNNParamInitializer.RECURRENT_WEIGHT_KEY:
			return l2;
		case RNNParamInitializer.BIAS_KEY:
			return l2Bias;
		default:
			throw new IllegalArgumentException("Unknown parameter name: \"" + paramName + "\"");
		}
	}

	@Override
	public LayerMemoryReport getMemoryReport(InputType inputType) {
		return null;
	}

	public static class Builder extends BaseRecurrentLayer.Builder<Builder> {

		@SuppressWarnings("unchecked")
		public RNN build() {
			return new RNN(this);
		}
	}
}
