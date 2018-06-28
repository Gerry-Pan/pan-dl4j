package personal.pan.dl4j.nn.graph;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.BaseGraphVertex;
import org.deeplearning4j.nn.graph.vertex.VertexIndices;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Or;
import org.nd4j.linalg.api.ops.impl.transforms.SoftMaxDerivative;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;

/**
 * 
 * @author Jerry
 *
 */
public class AttentionVertex extends BaseGraphVertex {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public AttentionVertex(ComputationGraph graph, String name, int vertexIndex) {
		super(graph, name, vertexIndex, null, null);
	}

	public AttentionVertex(ComputationGraph graph, String name, int vertexIndex, VertexIndices[] inputVertices,
			VertexIndices[] outputVertices) {
		super(graph, name, vertexIndex, inputVertices, outputVertices);
	}

	@Override
	public boolean hasLayer() {
		return false;
	}

	@Override
	public Layer getLayer() {
		return null;
	}

	@Override
	public INDArray doForward(boolean training) {
		if (!canDoForward())
			throw new IllegalStateException("Cannot do forward pass: inputs not set");

		if (inputs.length == 1) {
			// No-op case
			return inputs[0];
		}

		INDArray x1 = inputs[0];
		INDArray x2 = inputs[1];

		int[] fwdPassShape = x1.shape();
		int batchSize = fwdPassShape[0];
		int maxTsLength = fwdPassShape[2];

		INDArray[] inputMaskArrays = graph.getInputMaskArrays();
		INDArray firstMask = inputMaskArrays[0];
		INDArray secondMask = inputMaskArrays[1];

		INDArray row = Nd4j.linspace(0, maxTsLength - 1, maxTsLength);

		INDArray firstMaskArray = firstMask.mulRowVector(row);
		INDArray secondMaskArray = secondMask.mulRowVector(row);

		INDArray firstLastElementIdx = Nd4j.argMax(firstMaskArray, 1);
		INDArray secondLastElementIdx = Nd4j.argMax(secondMaskArray, 1);

		INDArray result = Nd4j.create(fwdPassShape);

		for (int i = 0; i < batchSize; i++) {
			int firstLastIdx = (int) firstLastElementIdx.getDouble(i);
			int secondLastIdx = (int) secondLastElementIdx.getDouble(i);

			INDArray q = x1.get(new INDArrayIndex[] { NDArrayIndex.point(i), NDArrayIndex.all(),
					NDArrayIndex.interval(0, firstLastIdx) });
			INDArray k = x2.get(new INDArrayIndex[] { NDArrayIndex.point(i), NDArrayIndex.all(),
					NDArrayIndex.interval(0, secondLastIdx) });

			INDArray r = attention(q, k);

			result.put(new INDArrayIndex[] { NDArrayIndex.point(i), NDArrayIndex.all(),
					NDArrayIndex.interval(0, secondLastIdx) }, r);
		}

		return result;
	}

	@Override
	public Pair<Gradient, INDArray[]> doBackward(boolean tbptt) {
		if (!canDoBackward())
			throw new IllegalStateException("Cannot do backward pass: errors not set");

		INDArray x1 = inputs[0];
		INDArray x2 = inputs[1];

		int[] fwdPassShape = x1.shape();
		int dk = fwdPassShape[1];
		int batchSize = fwdPassShape[0];
		int maxTsLength = fwdPassShape[2];

		INDArray[] inputMaskArrays = graph.getInputMaskArrays();
		INDArray firstMask = inputMaskArrays[0];
		INDArray secondMask = inputMaskArrays[1];

		INDArray row = Nd4j.linspace(0, maxTsLength - 1, maxTsLength);

		INDArray firstMaskArray = firstMask.mulRowVector(row);
		INDArray secondMaskArray = secondMask.mulRowVector(row);

		INDArray firstLastElementIdx = Nd4j.argMax(firstMaskArray, 1);
		INDArray secondLastElementIdx = Nd4j.argMax(secondMaskArray, 1);

		INDArray x1EpsilonNext = Nd4j.create(x1.shape());
		INDArray x2EpsilonNext = Nd4j.create(x2.shape());

		for (int i = 0; i < batchSize; i++) {
			int firstLastIdx = (int) firstLastElementIdx.getDouble(i);
			int secondLastIdx = (int) secondLastElementIdx.getDouble(i);

			INDArray q = x1.get(new INDArrayIndex[] { NDArrayIndex.point(i), NDArrayIndex.all(),
					NDArrayIndex.interval(0, firstLastIdx) });
			INDArray k = x2.get(new INDArrayIndex[] { NDArrayIndex.point(i), NDArrayIndex.all(),
					NDArrayIndex.interval(0, secondLastIdx) });

			INDArray alpha = k.transpose().mmul(q).div(Math.sqrt(dk));
			INDArray softMax = Transforms.softmax(alpha);
			INDArray softMaxDerivative = Nd4j.getExecutioner().exec(new SoftMaxDerivative(alpha)).z();

			INDArray e = epsilon.get(new INDArrayIndex[] { NDArrayIndex.point(i), NDArrayIndex.all(),
					NDArrayIndex.interval(0, secondLastIdx) });

			INDArray temp = q.transpose().mmul(e).mul(softMaxDerivative.transpose());

			INDArray dLdk = q.div(Math.sqrt(dk)).mmul(temp);
			INDArray dLdq = k.div(Math.sqrt(dk)).mmul(temp.transpose());

			Nd4j.gemm(e, softMax, dLdq, false, false, 1.0, 0.0);

			x1EpsilonNext.put(new INDArrayIndex[] { NDArrayIndex.point(i), NDArrayIndex.all(),
					NDArrayIndex.interval(0, firstLastIdx) }, dLdq);
			x2EpsilonNext.put(new INDArrayIndex[] { NDArrayIndex.point(i), NDArrayIndex.all(),
					NDArrayIndex.interval(0, secondLastIdx) }, dLdk);
		}

		INDArray[] epsilonNext = new INDArray[] { x1EpsilonNext, x2EpsilonNext };

		return new Pair<>(null, epsilonNext);
	}

	@Override
	public void setBackpropGradientsViewArray(INDArray backpropGradientsViewArray) {
		if (backpropGradientsViewArray != null)
			throw new RuntimeException("Vertex does not have gradients; gradients view array cannot be set here");
	}

	@Override
	public Pair<INDArray, MaskState> feedForwardMaskArrays(INDArray[] maskArrays, MaskState currentMaskState,
			int minibatchSize) {
		if (maskArrays == null) {
			return new Pair<>(null, currentMaskState);
		}

		for (INDArray arr : maskArrays) {
			if (arr == null) {
				return new Pair<>(null, currentMaskState);
			}
		}

		if (maskArrays.length == 1) {
			return new Pair<>(maskArrays[0], currentMaskState);
		} else {
			INDArray ret = maskArrays[0].dup(maskArrays[0].ordering());
			Nd4j.getExecutioner().exec(new Or(maskArrays[0], maskArrays[1], ret));
			for (int i = 2; i < maskArrays.length; i++) {
				Nd4j.getExecutioner().exec(new Or(maskArrays[i], ret, ret));
			}
			return new Pair<>(ret, currentMaskState);
		}
	}

	/**
	 * scaled dot-product attention
	 * 
	 * @param q
	 *            shape(w2vLayerSize, timeLenthForQ)
	 * @param k
	 *            shape(w2vLayerSize, timeLenthForK)
	 */
	protected INDArray attention(INDArray q, INDArray k) {
		int dk = k.rows();

		// shape(timeLenthForK, timeLenthForQ)
		INDArray alpha = k.transpose().mmul(q).div(Math.sqrt(dk));

		Transforms.softmax(alpha, false);

		return q.mmul(alpha.transpose());// shape(w2vLayerSize, timeLenthForK)
	}

	@Override
	public String toString() {
		return "AttentionVertex(id=" + this.getVertexIndex() + ",name=\"" + this.getVertexName() + "\"" + ")";
	}
}
