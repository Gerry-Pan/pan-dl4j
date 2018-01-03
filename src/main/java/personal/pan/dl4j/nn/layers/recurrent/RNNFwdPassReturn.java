package personal.pan.dl4j.nn.layers.recurrent;

import org.nd4j.linalg.api.ndarray.INDArray;

public class RNNFwdPassReturn {

	public INDArray fwdPassOutput;

	/**
	 * output activation,h_t=σ(X_t*W + h_t-1*U + b)
	 */
	public INDArray[] fwdPassOutputAsArrays;

	public INDArray prevAct;

	public INDArray lastAct;
}
