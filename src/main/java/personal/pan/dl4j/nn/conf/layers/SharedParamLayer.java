package personal.pan.dl4j.nn.conf.layers;

import java.util.Collection;

import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.graph.SubsetVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
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
 * 共用参数层，与{@link MergeVertex}和{@link SubsetVertex}结合使用 。<br>
 * 层与层共用参数，总损失值对该参数的梯度为，各层梯度的和。<br>
 * <br>
 * L_total=L_1 + L_2<br>
 * ∂L_total/∂θ=∂L_1/∂θ + ∂L_2/∂θ<br>
 * <br>
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

	private FeedForwardLayer layer;

	private SharedParamLayer(Builder builder) {
		super(builder);

		this.layer = builder.layer;
		this.inputCount = builder.inputCount;

		this.nIn = layer.getNIn();
		this.nOut = layer.getNOut();

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
		layer.setIUpdater(iUpdater);
	}

	@Override
	public IUpdater getIUpdater() {
		return layer.getIUpdater();
	}

	@Override
	public void setBiasUpdater(IUpdater biasUpdater) {
		layer.setBiasUpdater(biasUpdater);
	}

	@Override
	public IUpdater getBiasUpdater() {
		return layer.getBiasUpdater();
	}

	@Override
	public void setWeightInit(WeightInit weightInit) {
		layer.setWeightInit(weightInit);
	}

	@Override
	public void setBiasInit(double biasInit) {
		layer.setBiasInit(biasInit);
	}

	@Override
	public void setL1(double l1) {
		layer.setL1(l1);
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
		return layer.getActivationFn();
	}

	@Override
	public double getBiasInit() {
		return layer.getBiasInit();
	}

	@Override
	public Distribution getDist() {
		return layer.getDist();
	}

	@Override
	public double getL1() {
		return layer.getL1();
	}

	@Override
	public double getL1Bias() {
		return layer.getL1Bias();
	}

	@Override
	public double getL2() {
		return layer.getL2();
	}

	@Override
	public double getL2Bias() {
		return layer.getL2Bias();
	}

	@Override
	public WeightInit getWeightInit() {
		return layer.getWeightInit();
	}

	@Override
	public IWeightNoise getWeightNoise() {
		return layer.getWeightNoise();
	}

	@Override
	public void setActivationFn(IActivation activationFn) {
		layer.setActivationFn(activationFn);
	}

	@Override
	public void setDist(Distribution dist) {
		layer.setDist(dist);
	}

	@Override
	public void setGradientNormalization(GradientNormalization gradientNormalization) {
		layer.setGradientNormalization(gradientNormalization);
	}

	@Override
	public void setGradientNormalizationThreshold(double gradientNormalizationThreshold) {
		layer.setGradientNormalizationThreshold(gradientNormalizationThreshold);
	}

	@Override
	public void setL1Bias(double l1Bias) {
		layer.setL1Bias(l1Bias);
	}

	@Override
	public void setL2(double l2) {
		layer.setL2(l2);
	}

	@Override
	public void setL2Bias(double l2Bias) {
		layer.setL2Bias(l2Bias);
	}

	@Override
	public void setWeightNoise(IWeightNoise weightNoise) {
		layer.setWeightNoise(weightNoise);
	}

	@Override
	public IUpdater getUpdaterByParam(String paramName) {
		return layer.getUpdaterByParam(paramName);
	}

	@Override
	public InputType getOutputType(int layerIndex, InputType inputType) {
		return layer.getOutputType(layerIndex, inputType);
	}

	@Override
	public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
		return layer.getPreProcessorForInputType(inputType);
	}

	@Override
	public void setNIn(InputType inputType, boolean override) {
		layer.setNIn(inputType, override);
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

		private FeedForwardLayer layer;

		private long inputCount;

		public Builder inputCount(int inputCount) {
			this.inputCount = inputCount;
			return this;
		}

		public Builder layer(FeedForwardLayer layer) {
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
