package personal.pan.dl4j.nn.lossfunctions;

import java.util.List;
import java.util.Map;

import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossUtil;
import org.nd4j.linalg.lossfunctions.serde.RowVectorDeserializer;
import org.nd4j.linalg.lossfunctions.serde.RowVectorSerializer;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.shade.jackson.databind.annotation.JsonDeserialize;
import org.nd4j.shade.jackson.databind.annotation.JsonSerialize;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import onnx.OnnxProto3;
import personal.pan.dl4j.nn.activations.ActivationSiamese;

public class LossContrastiveL2 extends DifferentialFunction implements ILossFunction {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	private final double margin;

	@JsonSerialize(using = RowVectorSerializer.class)
	@JsonDeserialize(using = RowVectorDeserializer.class)
	protected final INDArray weights;

	public LossContrastiveL2(double margin) {
		this(margin, null);
	}

	public LossContrastiveL2(double margin, INDArray weights) {
		this.margin = margin;
		this.weights = weights;
	}

	/**
	 * preOutput=[X1,X2]<br>
	 * output=cosine(preOutput)<br>
	 * if output < margin,l=[(margin-output)^2,output^2] <br>
	 * if output > margin,l=[0,output^2]<br>
	 * l=[L−(X1,X2),L+(X1,X2)]<br>
	 * labels=[1-y,y]<br>
	 * L=Lw(X1,X2)=(1−y)L−(X1,X2)+yL+(X1,X2)<br>
	 * L=l.mul(labels)<br>
	 * 
	 * @param labels
	 *            shape(batchSize,2)
	 * @param preOutput
	 *            shape(batchSize,1)&nbsp;&nbsp;&nbsp;&nbsp;此参数为链接中的Ew,即cosine值
	 * @param activationFn
	 *            {@link ActivationSiamese}
	 * @param mask
	 * @return
	 */
	private INDArray scoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
		INDArray output = activationFn.getActivation(preOutput, true);

		INDArray outputPositive = output.mul(output).mul(2 / margin);
		INDArray outputNegative = Transforms.exp(output.mul(-2.77 / margin), true).mul(2 * margin);

		INDArray l = Nd4j.create(labels.shape());

		l.put(new INDArrayIndex[] { NDArrayIndex.all(), NDArrayIndex.point(0) }, outputNegative);
		l.put(new INDArrayIndex[] { NDArrayIndex.all(), NDArrayIndex.point(1) }, outputPositive);

		INDArray scoreArr = l.mul(labels);

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

		if (average) {
			score /= scoreArr.size(0);
		}

		return score;
	}

	@Override
	public INDArray computeScoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
		INDArray scoreArr = scoreArray(labels, preOutput, activationFn, mask);
		return scoreArr.sum(1);
	}

	@Override
	public INDArray computeGradient(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
		INDArray dLdl = labels.dup();

		INDArray output = activationFn.getActivation(preOutput, true);

		INDArray derivativeNegative = output.mul(4 / margin);
		INDArray derivativePositive = Transforms.exp(output.mul(-2.77 / margin), true).mul(-2 * 2.77);

		INDArray derivative = Nd4j.hstack(derivativeNegative, derivativePositive);

		INDArray dLda = derivative.mul(dLdl).sum(1);

		INDArray dLdoutput = activationFn.backprop(preOutput, dLda).getFirst();

		return dLdoutput;
	}

	@Override
	public Pair<Double, INDArray> computeGradientAndScore(INDArray labels, INDArray preOutput, IActivation activationFn,
			INDArray mask, boolean average) {
		return new Pair<>(computeScore(labels, preOutput, activationFn, mask, average),
				computeGradient(labels, preOutput, activationFn, mask));
	}

	/**
	 * The opName of this function
	 *
	 * @return
	 */
	@Override
	public String name() {
		return toString();
	}

	@Override
	public String toString() {
		if (weights == null)
			return "LossContrastiveL2()";
		return "LossContrastiveL2(weights=" + weights + ")";
	}

	@Override
	public SDVariable[] outputVariables() {
		return new SDVariable[0];
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
	public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith,
			Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {

	}

	@Override
	public String onnxName() {
		return "ContrastiveL2Loss";
	}

	@Override
	public String tensorflowName() {
		return "ContrastiveL2Loss";
	}

}
