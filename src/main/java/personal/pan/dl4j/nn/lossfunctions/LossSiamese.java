package personal.pan.dl4j.nn.lossfunctions;

import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.impl.LossMCXENT;

public class LossSiamese extends LossMCXENT {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	@Override
	public INDArray computeGradient(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
		// do it in future
		return super.computeGradient(labels, preOutput, activationFn, mask);
	}
}
