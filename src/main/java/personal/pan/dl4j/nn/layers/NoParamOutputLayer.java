package personal.pan.dl4j.nn.layers;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseOutputLayer;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.primitives.Pair;

public class NoParamOutputLayer extends BaseOutputLayer<personal.pan.dl4j.nn.conf.layers.NoParamOutputLayer> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public NoParamOutputLayer(NeuralNetConfiguration conf) {
		super(conf);
	}

	public NoParamOutputLayer(NeuralNetConfiguration conf, INDArray input) {
		super(conf, input);
	}

	@Override
	protected INDArray getLabels2d(LayerWorkspaceMgr workspaceMgr, ArrayType arrayType) {
		return labels;
	}

	@Override
	protected INDArray preOutput(boolean training, LayerWorkspaceMgr workspaceMgr) {
		return this.input;
	}

	@Override
	public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
		ILossFunction lossFunction = layerConf().getLossFn();
		INDArray labels2d = getLabels2d(workspaceMgr, ArrayType.BP_WORKING_MEM);
		INDArray epsilonNext = lossFunction.computeGradient(labels2d, preOutput2d(true, workspaceMgr),
				layerConf().getActivationFn(), maskArray);

		epsilonNext = workspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD, epsilonNext);

		epsilonNext = backpropDropOutIfPresent(epsilonNext);
		return new Pair<Gradient, INDArray>(null, epsilonNext);
	}
}
