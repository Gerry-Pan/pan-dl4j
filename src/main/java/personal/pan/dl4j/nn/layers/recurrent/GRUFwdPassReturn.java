package personal.pan.dl4j.nn.layers.recurrent;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * created on 2017-12-29
 * 
 * @author Gerry Pan
 *
 */
public class GRUFwdPassReturn {

	/**
	 * output activation of the last time
	 */
	public INDArray fwdPassOutput;

	/**
	 * output activation,h_t=(1-z_t)⊙h_t-1 + z_t⊙ĥ_t
	 */
	public INDArray[] fwdPassOutputAsArrays;

	/**
	 * update gate
	 */
	public INDArray[] uz;

	/**
	 * update gate activation,z_t=σ(X_t*W + h_t-1*U)
	 */
	public INDArray[] ua;

	/**
	 * reset gate
	 */
	public INDArray[] rz;

	/**
	 * reset gate activation,r_t=σ(X_t*W + h_t-1*U)
	 */
	public INDArray[] ra;

	/**
	 * candidate
	 */
	public INDArray[] hz;

	/**
	 * candidate activation,ĥ_t=tanh(X_t*W + (r_t⊙h_t-1)*U)
	 */
	public INDArray[] ha;

	public INDArray prevAct;

	public INDArray lastAct;
}
