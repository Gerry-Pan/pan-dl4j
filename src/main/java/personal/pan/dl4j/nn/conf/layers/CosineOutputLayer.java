package personal.pan.dl4j.nn.conf.layers;

import java.util.Collection;
import java.util.Map;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.BaseOutputLayer;
import org.deeplearning4j.nn.conf.layers.LayerValidation;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import lombok.NoArgsConstructor;

public class CosineOutputLayer extends BaseOutputLayer {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	protected CosineOutputLayer(Builder builder) {
		super(builder);
	}

	@Override
	public Layer instantiate(NeuralNetConfiguration conf, Collection<IterationListener> iterationListeners,
			int layerIndex, INDArray layerParamsView, boolean initializeParams) {
		LayerValidation.assertNInNOutSet("OutputLayer", getLayerName(), layerIndex, getNIn(), getNOut());

		personal.pan.dl4j.nn.layers.CosineOutputLayer ret = new personal.pan.dl4j.nn.layers.CosineOutputLayer(conf);
		ret.setListeners(iterationListeners);
		ret.setIndex(layerIndex);
		ret.setParamsViewArray(layerParamsView);
		Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
		ret.setParamTable(paramTable);
		ret.setConf(conf);
		return ret;
	}

	@Override
	public ParamInitializer initializer() {
		return DefaultParamInitializer.getInstance();
	}

	@NoArgsConstructor
	public static class Builder extends BaseOutputLayer.Builder<Builder> {

		public Builder(LossFunction lossFunction) {
			super.lossFunction(lossFunction);
		}

		public Builder(ILossFunction lossFunction) {
			this.lossFn = lossFunction;
		}

		@Override
		@SuppressWarnings("unchecked")
		public CosineOutputLayer build() {
			return new CosineOutputLayer(this);
		}
	}
}
