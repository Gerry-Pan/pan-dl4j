package personal.pan.dl4j.nn.conf.layers;

import java.util.Collection;
import java.util.List;
import java.util.Map;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.BaseOutputLayer;
import org.deeplearning4j.nn.conf.layers.LayerValidation;
import org.deeplearning4j.nn.params.EmptyParamInitializer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.regularization.Regularization;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.ToString;

@Data
@NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class NoParamOutputLayer extends BaseOutputLayer {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	protected NoParamOutputLayer(Builder builder) {
		super(builder);
		initializeConstraints(builder);
	}

	@Override
	public Layer instantiate(NeuralNetConfiguration conf, Collection<TrainingListener> trainingListeners,
			int layerIndex, INDArray layerParamsView, boolean initializeParams, DataType networkDatatype) {
		LayerValidation.assertNInNOutSet("NoParamOutputLayer", getLayerName(), layerIndex, getNIn(), getNOut());

		personal.pan.dl4j.nn.layers.NoParamOutputLayer ret = new personal.pan.dl4j.nn.layers.NoParamOutputLayer(conf,
				networkDatatype);
		ret.setListeners(trainingListeners);
		ret.setIndex(layerIndex);
		ret.setParamsViewArray(layerParamsView);
		Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
		ret.setParamTable(paramTable);
		ret.setConf(conf);
		return ret;
	}

	@Override
	public ParamInitializer initializer() {
		return EmptyParamInitializer.getInstance();
	}

	@Override
	public List<Regularization> getRegularizationByParam(String paramName) {
		return null;
	}

	@Override
	public GradientNormalization getGradientNormalization() {
		return GradientNormalization.None;
	}

	@Override
	public double getGradientNormalizationThreshold() {
		return 0;
	}

	@Override
	public boolean isPretrainParam(String paramName) {
		throw new UnsupportedOperationException(getClass().getSimpleName() + " does not contain parameters");
	}

	public static class Builder extends BaseOutputLayer.Builder<Builder> {

		public Builder() {
			this.activationFn = new ActivationSoftmax();
		}

		public Builder(LossFunction lossFunction) {
			super.lossFunction(lossFunction);
			this.activationFn = new ActivationSoftmax();
		}

		public Builder(ILossFunction lossFunction) {
			this.lossFn = lossFunction;
			this.activationFn = new ActivationSoftmax();
		}

		@Override
		@SuppressWarnings("unchecked")
		public NoParamOutputLayer build() {
			return new NoParamOutputLayer(this);
		}
	}
}
