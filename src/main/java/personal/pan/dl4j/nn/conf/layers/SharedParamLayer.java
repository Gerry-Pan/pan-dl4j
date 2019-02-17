package personal.pan.dl4j.nn.conf.layers;

import java.util.Collection;

import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.weightnoise.IWeightNoise;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.config.IUpdater;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.ToString;
import personal.pan.dl4j.nn.params.SharedParamLayerParamInitializer;

/**
 * 2019-2-16
 * 
 * @author Jerry Pan
 *
 */
@Data
@NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class SharedParamLayer extends FeedForwardLayer {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	private long inputCount;

	private Layer layer;

	private SharedParamLayer(Builder builder) {
		super(builder);

		this.layer = builder.layer;
		this.inputCount = builder.inputCount;

		if (layer instanceof FeedForwardLayer) {
			FeedForwardLayer ffLayer = (FeedForwardLayer) layer;

			this.nIn = ffLayer.getNIn();
			this.nOut = ffLayer.getNOut();
		}

		initializeConstraints(builder);
	}

	@Override
	public org.deeplearning4j.nn.api.Layer instantiate(NeuralNetConfiguration conf,
			Collection<TrainingListener> trainingListeners, int layerIndex, INDArray layerParamsView,
			boolean initializeParams) {
		personal.pan.dl4j.nn.layers.SharedParamLayer sharedParamLayer = new personal.pan.dl4j.nn.layers.SharedParamLayer(
				conf);

		org.deeplearning4j.nn.api.Layer l = layer.instantiate(getInnerConf(conf), trainingListeners, layerIndex,
				layerParamsView, initializeParams);

		sharedParamLayer.setLayer(l);
		sharedParamLayer.setInputCount(this.inputCount);
		sharedParamLayer.setListeners(trainingListeners);
		sharedParamLayer.setIndex(layerIndex);
		sharedParamLayer.setParamsViewArray(layerParamsView);
		sharedParamLayer.setParamTable(l.paramTable());
		sharedParamLayer.setConf(conf);
		return sharedParamLayer;
	}

	@Override
	public ParamInitializer initializer() {
		return SharedParamLayerParamInitializer.getInstance();
	}

	@Override
	public LayerMemoryReport getMemoryReport(InputType inputType) {
		return layer.getMemoryReport(inputType);
	}

	@Override
	public double getL1ByParam(String paramName) {
		return layer.getL1ByParam(paramName);
	}

	@Override
	public double getL2ByParam(String paramName) {
		return layer.getL2ByParam(paramName);
	}

	@Override
	public boolean isPretrainParam(String paramName) {
		return layer.isPretrainParam(paramName);
	}

	@Override
	public void setIUpdater(IUpdater iUpdater) {
		((org.deeplearning4j.nn.conf.layers.BaseLayer) layer).setIUpdater(iUpdater);
	}

	@Override
	public IUpdater getIUpdater() {
		return ((org.deeplearning4j.nn.conf.layers.BaseLayer) layer).getIUpdater();
	}

	@Override
	public void setBiasUpdater(IUpdater biasUpdater) {
		((org.deeplearning4j.nn.conf.layers.BaseLayer) layer).setBiasUpdater(biasUpdater);
	}

	@Override
	public IUpdater getBiasUpdater() {
		return ((org.deeplearning4j.nn.conf.layers.BaseLayer) layer).getBiasUpdater();
	}

	@Override
	public void setWeightInit(WeightInit weightInit) {
		((org.deeplearning4j.nn.conf.layers.BaseLayer) layer).setWeightInit(weightInit);
	}

	@Override
	public void setBiasInit(double biasInit) {
		((org.deeplearning4j.nn.conf.layers.BaseLayer) layer).setBiasInit(biasInit);
	}

	@Override
	public void setL1(double l1) {
		((org.deeplearning4j.nn.conf.layers.BaseLayer) layer).setL1(l1);
	}

	@Override
	public GradientNormalization getGradientNormalization() {
		return layer.getGradientNormalization();
	}

	@Override
	public double getGradientNormalizationThreshold() {
		return layer.getGradientNormalizationThreshold();
	}

	@Override
	public boolean isPretrain() {
		return layer.isPretrain();
	}

	@Override
	public void setLayerName(String layerName) {
		super.setLayerName(layerName);
		layer.setLayerName(layerName);
	}

	@Override
	public IActivation getActivationFn() {
		return ((org.deeplearning4j.nn.conf.layers.BaseLayer) layer).getActivationFn();
	}

	@Override
	public double getBiasInit() {
		return ((org.deeplearning4j.nn.conf.layers.BaseLayer) layer).getBiasInit();
	}

	@Override
	public Distribution getDist() {
		return ((org.deeplearning4j.nn.conf.layers.BaseLayer) layer).getDist();
	}

	@Override
	public double getL1() {
		return ((org.deeplearning4j.nn.conf.layers.BaseLayer) layer).getL1();
	}

	@Override
	public double getL1Bias() {
		return ((org.deeplearning4j.nn.conf.layers.BaseLayer) layer).getL1Bias();
	}

	@Override
	public double getL2() {
		return ((org.deeplearning4j.nn.conf.layers.BaseLayer) layer).getL2();
	}

	@Override
	public double getL2Bias() {
		return ((org.deeplearning4j.nn.conf.layers.BaseLayer) layer).getL2Bias();
	}

	@Override
	public WeightInit getWeightInit() {
		return ((org.deeplearning4j.nn.conf.layers.BaseLayer) layer).getWeightInit();
	}

	@Override
	public IWeightNoise getWeightNoise() {
		return ((org.deeplearning4j.nn.conf.layers.BaseLayer) layer).getWeightNoise();
	}

	@Override
	public void setActivationFn(IActivation activationFn) {
		((org.deeplearning4j.nn.conf.layers.BaseLayer) layer).setActivationFn(activationFn);
	}

	@Override
	public void setDist(Distribution dist) {
		((org.deeplearning4j.nn.conf.layers.BaseLayer) layer).setDist(dist);
	}

	@Override
	public void setGradientNormalization(GradientNormalization gradientNormalization) {
		((org.deeplearning4j.nn.conf.layers.BaseLayer) layer).setGradientNormalization(gradientNormalization);
	}

	@Override
	public void setGradientNormalizationThreshold(double gradientNormalizationThreshold) {
		((org.deeplearning4j.nn.conf.layers.BaseLayer) layer)
				.setGradientNormalizationThreshold(gradientNormalizationThreshold);
	}

	@Override
	public void setL1Bias(double l1Bias) {
		((org.deeplearning4j.nn.conf.layers.BaseLayer) layer).setL1Bias(l1Bias);
	}

	@Override
	public void setL2(double l2) {
		((org.deeplearning4j.nn.conf.layers.BaseLayer) layer).setL2(l2);
	}

	@Override
	public void setL2Bias(double l2Bias) {
		((org.deeplearning4j.nn.conf.layers.BaseLayer) layer).setL2Bias(l2Bias);
	}

	@Override
	public void setWeightNoise(IWeightNoise weightNoise) {
		((org.deeplearning4j.nn.conf.layers.BaseLayer) layer).setWeightNoise(weightNoise);
	}

	@Override
	public IUpdater getUpdaterByParam(String paramName) {
		return ((org.deeplearning4j.nn.conf.layers.BaseLayer) layer).getUpdaterByParam(paramName);
	}

	@Override
	public InputType getOutputType(int layerIndex, InputType inputType) {
		return ((org.deeplearning4j.nn.conf.layers.BaseLayer) layer).getOutputType(layerIndex, inputType);
	}

	@Override
	public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
		return ((org.deeplearning4j.nn.conf.layers.BaseLayer) layer).getPreProcessorForInputType(inputType);
	}

	@Override
	public void setNIn(InputType inputType, boolean override) {
		((org.deeplearning4j.nn.conf.layers.BaseLayer) layer).setNIn(inputType, override);
	}

	public long inputCount() {
		return inputCount;
	}

	public NeuralNetConfiguration getInnerConf(NeuralNetConfiguration conf) {
		NeuralNetConfiguration nnc = conf.clone();
		nnc.setLayer(layer);
		return nnc;
	}

	@NoArgsConstructor
	public static class Builder extends FeedForwardLayer.Builder<Builder> {

		private Layer layer;

		private long inputCount;

		public Builder inputCount(int inputCount) {
			this.inputCount = inputCount;
			return this;
		}

		public Builder layer(Layer layer) {
			this.layer = layer;
			return this;
		}

		@Override
		@SuppressWarnings("unchecked")
		public SharedParamLayer build() {
			return new SharedParamLayer(this);
		}
	}

}
