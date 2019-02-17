package personal.pan.dl4j.nn.params;

import java.util.List;
import java.util.Map;

import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.nd4j.linalg.api.ndarray.INDArray;

import personal.pan.dl4j.nn.conf.layers.SharedParamLayer;

public class SharedParamLayerParamInitializer implements ParamInitializer {

	private static final SharedParamLayerParamInitializer INSTANCE = new SharedParamLayerParamInitializer();

	public static SharedParamLayerParamInitializer getInstance() {
		return INSTANCE;
	}

	@Override
	public long numParams(NeuralNetConfiguration conf) {
		return numParams(conf.getLayer());
	}

	@Override
	public long numParams(Layer layer) {
		SharedParamLayer fl = (SharedParamLayer) layer;
		ParamInitializer initializer = fl.getLayer().initializer();
		return initializer.numParams(fl.getLayer());
	}

	@Override
	public List<String> paramKeys(Layer layer) {
		SharedParamLayer fl = (SharedParamLayer) layer;
		ParamInitializer initializer = fl.getLayer().initializer();
		return initializer.paramKeys(fl.getLayer());
	}

	@Override
	public List<String> weightKeys(Layer layer) {
		SharedParamLayer fl = (SharedParamLayer) layer;
		ParamInitializer initializer = fl.getLayer().initializer();
		return initializer.weightKeys(fl.getLayer());
	}

	@Override
	public List<String> biasKeys(Layer layer) {
		SharedParamLayer fl = (SharedParamLayer) layer;
		ParamInitializer initializer = fl.getLayer().initializer();
		return initializer.biasKeys(fl.getLayer());
	}

	@Override
	public boolean isWeightParam(Layer layer, String key) {
		SharedParamLayer fl = (SharedParamLayer) layer;
		ParamInitializer initializer = fl.getLayer().initializer();
		return initializer.isWeightParam(fl.getLayer(), key);
	}

	@Override
	public boolean isBiasParam(Layer layer, String key) {
		SharedParamLayer fl = (SharedParamLayer) layer;
		ParamInitializer initializer = fl.getLayer().initializer();
		return initializer.isBiasParam(fl.getLayer(), key);
	}

	@Override
	public Map<String, INDArray> init(NeuralNetConfiguration conf, INDArray paramsView, boolean initializeParams) {
		SharedParamLayer fl = (SharedParamLayer) conf.getLayer();
		Layer innerLayer = fl.getLayer();
		ParamInitializer initializer = innerLayer.initializer();
		conf.setLayer(innerLayer);
		Map<String, INDArray> m = initializer.init(conf, paramsView, initializeParams);
		conf.setLayer(fl);

		return m;
	}

	@Override
	public Map<String, INDArray> getGradientsFromFlattened(NeuralNetConfiguration conf, INDArray gradientView) {
		SharedParamLayer fl = (SharedParamLayer) conf.getLayer();
		Layer innerLayer = fl.getLayer();
		ParamInitializer initializer = innerLayer.initializer();
		conf.setLayer(innerLayer);
		Map<String, INDArray> m = initializer.getGradientsFromFlattened(conf, gradientView);
		conf.setLayer(fl);
		return m;
	}
}
