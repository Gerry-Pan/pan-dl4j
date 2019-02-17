package personal.pan.dl4j.nn.lossfunctions;

import java.util.List;
import java.util.Map;

import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossUtil;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import onnx.OnnxProto3.AttributeProto;
import onnx.OnnxProto3.GraphProto;
import onnx.OnnxProto3.NodeProto;

public class LossLogCosh extends DifferentialFunction implements ILossFunction {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	protected INDArray scoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
		if (!labels.equalShapes(preOutput)) {
			Preconditions.throwEx("Labels and preOutput must have equal shapes: got shapes %s vs %s", labels.shape(),
					preOutput.shape());
		}
		INDArray output = activationFn.getActivation(preOutput.dup(), true);
		INDArray temp = output.rsubi(labels);

		INDArray scoreArr = Transforms.log(Transforms.cosh(temp));

		if (mask != null) {
			LossUtil.applyMask(scoreArr, mask);
		}

		return scoreArr;
	}

	@Override
	public double computeScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask,
			boolean average) {
		INDArray scoreArr = scoreArray(labels, preOutput, activationFn, mask);

		double score = scoreArr.sumNumber().doubleValue();

		if (average)
			score /= scoreArr.size(0);

		return score;
	}

	@Override
	public INDArray computeScoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
		INDArray scoreArr = scoreArray(labels, preOutput, activationFn, mask);
		return scoreArr.sum(1);
	}

	@Override
	public INDArray computeGradient(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
		if (!labels.equalShapes(preOutput)) {
			Preconditions.throwEx("Labels and preOutput must have equal shapes: got shapes %s vs %s", labels.shape(),
					preOutput.shape());
		}

		INDArray output = activationFn.getActivation(preOutput.dup(), true);

		INDArray temp = output.rsubi(labels);
		INDArray dLda = Transforms.tanh(temp).mul(-1);

		if (mask != null && LossUtil.isPerOutputMasking(dLda, mask)) {
			LossUtil.applyMask(dLda, mask);
		}

		INDArray gradients = activationFn.backprop(preOutput, dLda).getFirst();

		if (mask != null) {
			LossUtil.applyMask(gradients, mask);
		}

		return gradients;
	}

	@Override
	public Pair<Double, INDArray> computeGradientAndScore(INDArray labels, INDArray preOutput, IActivation activationFn,
			INDArray mask, boolean average) {
		return new Pair<>(computeScore(labels, preOutput, activationFn, mask, average),
				computeGradient(labels, preOutput, activationFn, mask));
	}

	@Override
	public String name() {
		return toString();
	}

	@Override
	public String toString() {
		return "LossLogCosh()";
	}

	@Override
	public SDVariable[] outputVariables(String baseName) {
		return new SDVariable[0];
	}

	@Override
	public List<SDVariable> doDiff(List<SDVariable> f1) {
		return null;
	}

	@Override
	public String opName() {
		return name();
	}

	@Override
	public Op.Type opType() {
		return Op.Type.CUSTOM;
	}

	@Override
	public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode,
			GraphDef graph) {

	}

	@Override
	public void initFromOnnx(NodeProto node, SameDiff initWith, Map<String, AttributeProto> attributesForNode,
			GraphProto graph) {

	}

	@Override
	public String onnxName() {
		return "LogCoshLoss";
	}

	@Override
	public String tensorflowName() {
		return "LogCoshLoss";
	}

}
