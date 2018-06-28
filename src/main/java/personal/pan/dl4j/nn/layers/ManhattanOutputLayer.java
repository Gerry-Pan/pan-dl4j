package personal.pan.dl4j.nn.layers;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseOutputLayer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;

public class ManhattanOutputLayer extends BaseOutputLayer<personal.pan.dl4j.nn.conf.layers.ManhattanOutputLayer> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public ManhattanOutputLayer(NeuralNetConfiguration conf) {
		super(conf);
	}

	public ManhattanOutputLayer(NeuralNetConfiguration conf, INDArray input) {
		super(conf, input);
	}

	/**
	 * preOutput=z=-|x1-x2|
	 */
	@Override
	public INDArray preOutput(boolean training) {
		int columns = input.shape()[1];

		INDArray x1 = input.get(new INDArrayIndex[] { NDArrayIndex.all(), NDArrayIndex.interval(0, columns / 2) });
		INDArray x2 = input
				.get(new INDArrayIndex[] { NDArrayIndex.all(), NDArrayIndex.interval(columns / 2, columns) });

		INDArray preOut = x1.sub(x2).norm1(1);

		return preOut.mul(-1);
	}

	@Override
	public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon) {

		ILossFunction lossFunction = layerConf().getLossFn();

		INDArray labels2d = getLabels2d();
		INDArray preOutput = preOutput2d(true);

		// delta = dL/dz
		INDArray delta = lossFunction.computeGradient(labels2d, preOutput, layerConf().getActivationFn(), maskArray);

		int columns = input.shape()[1];

		INDArray x1 = input.get(new INDArrayIndex[] { NDArrayIndex.all(), NDArrayIndex.interval(0, columns / 2) });
		INDArray x2 = input
				.get(new INDArrayIndex[] { NDArrayIndex.all(), NDArrayIndex.interval(columns / 2, columns) });

		INDArray temp = x1.sub(x2);

		INDArray x1EpsilonNext = Transforms.sign(temp);
		INDArray x2EpsilonNext = Transforms.sign(temp).mul(-1);

		INDArray epsilonNext = Nd4j.hstack(x1EpsilonNext, x2EpsilonNext);
		epsilonNext.muliColumnVector(delta);

		return new Pair<Gradient, INDArray>(null, epsilonNext);
	}
}
