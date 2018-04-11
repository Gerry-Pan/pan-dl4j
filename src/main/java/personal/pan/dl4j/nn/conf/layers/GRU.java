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
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationSigmoid;
import org.nd4j.linalg.api.ndarray.INDArray;

import personal.pan.dl4j.nn.layers.recurrent.GRUHelpers;
import personal.pan.dl4j.nn.params.GRUParamInitializer;

/**
 * created on 2017-12-29
 * 
 * @author Gerry Pan
 *
 */
public class GRU extends BaseRecurrentLayer {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	private IActivation gateActivationFn = new ActivationSigmoid();

	public GRU() {
		super();
	}

	public GRU(Builder builder) {
		super(builder);
		this.gateActivationFn = builder.gateActivationFn;
	}

	@Override
	public Layer instantiate(NeuralNetConfiguration conf, Collection<IterationListener> iterationListeners,
			int layerIndex, INDArray layerParamsView, boolean initializeParams) {
		personal.pan.dl4j.nn.layers.recurrent.GRU layer = new personal.pan.dl4j.nn.layers.recurrent.GRU(conf);
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
		return GRUParamInitializer.INSTANCE;
	}

	@Override
	public double getL1ByParam(String paramName) {
		switch (paramName) {
		case GRUParamInitializer.INPUT_WEIGHT_KEY:
		case GRUParamInitializer.RECURRENT_WEIGHT_KEY:
			return l1;
		case GRUParamInitializer.BIAS_KEY:
			return l1Bias;
		default:
			throw new IllegalArgumentException("Unknown parameter name: \"" + paramName + "\"");
		}
	}

	@Override
	public double getL2ByParam(String paramName) {
		switch (paramName) {
		case GRUParamInitializer.INPUT_WEIGHT_KEY:
		case GRUParamInitializer.RECURRENT_WEIGHT_KEY:
			return l2;
		case GRUParamInitializer.BIAS_KEY:
			return l2Bias;
		default:
			throw new IllegalArgumentException("Unknown parameter name: \"" + paramName + "\"");
		}
	}

	@Override
	public LayerMemoryReport getMemoryReport(InputType inputType) {
		return GRUHelpers.getMemoryReport(this, inputType);
	}

	public IActivation getGateActivationFn() {
		return gateActivationFn;
	}

	public void setGateActivationFn(IActivation gateActivationFn) {
		this.gateActivationFn = gateActivationFn;
	}

	public static class Builder extends BaseRecurrentLayer.Builder<Builder> {

		protected IActivation gateActivationFn = new ActivationSigmoid();

		@SuppressWarnings("unchecked")
		public GRU build() {
			return new GRU(this);
		}

	}
}
