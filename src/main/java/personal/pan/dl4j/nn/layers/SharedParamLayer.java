package personal.pan.dl4j.nn.layers;

import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.shade.protobuf.common.primitives.Ints;

import lombok.Getter;
import lombok.Setter;

/**
 * 2019-2-16
 * 
 * @author Jerry Pan
 *
 */
public class SharedParamLayer extends BaseLayer<personal.pan.dl4j.nn.conf.layers.SharedParamLayer> {

	@Setter
	@Getter
	private long inputCount;

	@Setter
	private int[] gradientIndexs;

	@Setter
	protected Layer layer;

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public SharedParamLayer(NeuralNetConfiguration conf, DataType dataType) {
		super(conf, dataType);
	}

	@Override
	public boolean isPretrainLayer() {
		return layer.isPretrainLayer();
	}

	@Override
	public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {
		List<INDArray> inputList = split(this.input);
		return activateInternal(inputList, training, workspaceMgr);
	}

	@Override
	public INDArray activate(INDArray input, boolean training, LayerWorkspaceMgr workspaceMgr) {
		List<INDArray> inputList = split(input);
		return activateInternal(inputList, training, workspaceMgr);
	}

	protected INDArray activateInternal(List<INDArray> inputList, boolean training, LayerWorkspaceMgr workspaceMgr) {
		List<INDArray> outputList = new LinkedList<INDArray>();
		for (INDArray input : inputList) {
			INDArray output = layer.activate(input, training, workspaceMgr);

			outputList.add(output);
		}

		return workspaceMgr.leverageTo(ArrayType.ACTIVATIONS,
				Nd4j.hstack(outputList.toArray(new INDArray[outputList.size()])));
	}

	@Override
	public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
		Map<String, INDArray> gm = null;
		Gradient gradient = new DefaultGradient();
		List<INDArray> epsilonList = split(epsilon);
		List<INDArray> epsilonNextList = new LinkedList<INDArray>();

		List<Integer> gradientIndexs = null;

		if (this.gradientIndexs == null || this.gradientIndexs.length == 0) {
			gradientIndexs = new LinkedList<Integer>();
			for (int i = 0; i < epsilonList.size(); i++) {
				gradientIndexs.add(i);
			}
		} else {
			gradientIndexs = Ints.asList(this.gradientIndexs);
		}

		for (int i = 0; i < epsilonList.size(); i++) {
			INDArray e = epsilonList.get(i);
			Pair<Gradient, INDArray> pair = layer.backpropGradient(e, workspaceMgr);

			INDArray epsilonNext = pair.getSecond();
			epsilonNextList.add(epsilonNext);

			if (!gradientIndexs.contains(i)) {
				continue;
			}

			Gradient singleGradient = pair.getFirst();

			gm = singleGradient.gradientForVariable();

			Set<String> keySet = gm.keySet();
			for (String key : keySet) {
				INDArray g1 = gm.get(key);
				INDArray g2 = gradient.getGradientFor(key);

				if (g2 == null) {
					g2 = Nd4j.zeros(g1.shape(), g1.ordering());
				}

				gradient.setGradientFor(key, g2.addi(g1));
			}
		}

		return new Pair<>(gradient, workspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD,
				Nd4j.hstack(epsilonNextList.toArray(new INDArray[epsilonNextList.size()]))));
	}

	@Override
	public void update(Gradient gradient) {
		layer.update(gradient);
	}

	@Override
	public void update(INDArray gradient, String paramType) {
		layer.update(gradient, paramType);
	}

	@Override
	public void setBackpropGradientsViewArray(INDArray gradients) {
		layer.setBackpropGradientsViewArray(gradients);
	}

	@Override
	public Map<String, INDArray> paramTable() {
		return layer.paramTable();
	}

	@Override
	public INDArray params() {
		return layer.params();
	}

	@Override
	public Map<String, INDArray> paramTable(boolean backpropParamsOnly) {
		return layer.paramTable(backpropParamsOnly);
	}

	@Override
	public void setParam(String key, INDArray val) {
		layer.setParam(key, val);
	}

	@Override
	public void setParams(INDArray params) {
		layer.setParams(params);
	}

	@Override
	public void setParamsViewArray(INDArray params) {
		layer.setParamsViewArray(params);
	}

	@Override
	public void setParamTable(Map<String, INDArray> paramTable) {
		layer.setParamTable(paramTable);
	}

	@Override
	public long numParams() {
		return layer.numParams();
	}

	@Override
	public long numParams(boolean backwards) {
		return layer.numParams(backwards);
	}

	/**
	 * 按照inputCount数量拆分张量
	 * 
	 * @param tensor
	 * @return
	 */
	protected List<INDArray> split(INDArray tensor) {
		List<INDArray> tensorList = new LinkedList<INDArray>();

		int rank = tensor.rank();

		if (rank != 2 && rank != 3 && rank != 4) {
			throw new RuntimeException("Illegal shape of tensor.");
		}

		long[] shape = tensor.shape();

		if (shape[1] % inputCount != 0) {
			throw new RuntimeException("Illegal shape of tensor.");
		}

		long n = shape[1] / inputCount;

		for (int i = 0; i < inputCount; i++) {
			INDArray in = null;

			if (rank == 2) {
				in = tensor.get(NDArrayIndex.all(), NDArrayIndex.interval(i * n, (i + 1) * n)).dup();
			}

			if (rank == 3) {
				in = tensor.get(NDArrayIndex.all(), NDArrayIndex.interval(i * n, (i + 1) * n), NDArrayIndex.all())
						.dup();
			}

			if (rank == 4) {
				in = tensor.get(NDArrayIndex.all(), NDArrayIndex.interval(i * n, (i + 1) * n), NDArrayIndex.all(),
						NDArrayIndex.all()).dup();
			}

			tensorList.add(in);
		}

		return tensorList;
	}

}
