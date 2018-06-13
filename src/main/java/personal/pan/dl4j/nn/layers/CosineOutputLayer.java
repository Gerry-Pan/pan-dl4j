package personal.pan.dl4j.nn.layers;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseOutputLayer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.primitives.Pair;

/**
 * 
 * @author Jerry
 *
 */
public class CosineOutputLayer extends BaseOutputLayer<personal.pan.dl4j.nn.conf.layers.CosineOutputLayer> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public CosineOutputLayer(NeuralNetConfiguration conf) {
		super(conf);
	}

	public CosineOutputLayer(NeuralNetConfiguration conf, INDArray input) {
		super(conf, input);
	}

	/**
	 * preOutput=input=[X1,X2]
	 */
	@Override
	public INDArray preOutput(boolean training) {
		return input;
	}

	@Override
	public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon) {

		ILossFunction lossFunction = layerConf().getLossFn();

		INDArray labels2d = getLabels2d();
		INDArray preOutput = preOutput2d(true);

		INDArray delta = lossFunction.computeGradient(labels2d, preOutput, layerConf().getActivationFn(), maskArray);

		INDArray epsilonNext = delta;

		return new Pair<Gradient, INDArray>(null, epsilonNext);
	}
}
