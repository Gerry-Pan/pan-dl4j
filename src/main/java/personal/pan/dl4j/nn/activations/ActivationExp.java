package personal.pan.dl4j.nn.activations;

import org.nd4j.linalg.activations.BaseActivationFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;

public class ActivationExp extends BaseActivationFunction {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	@Override
	public INDArray getActivation(INDArray in, boolean training) {
		return Transforms.exp(in, true);
	}

	@Override
	public Pair<INDArray, INDArray> backprop(INDArray in, INDArray epsilon) {
		INDArray epsilonNext = Transforms.exp(in, true);
		return new Pair<>(epsilonNext, null);
	}

}
