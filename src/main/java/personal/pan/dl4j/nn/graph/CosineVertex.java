package personal.pan.dl4j.nn.graph;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.BaseGraphVertex;
import org.deeplearning4j.nn.graph.vertex.VertexIndices;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Or;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;

public class CosineVertex extends BaseGraphVertex {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public CosineVertex(ComputationGraph graph, String name, int vertexIndex) {
		super(graph, name, vertexIndex, null, null);
	}

	public CosineVertex(ComputationGraph graph, String name, int vertexIndex, VertexIndices[] inputVertices,
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

		INDArray x1mag = x1.norm2(1);
		INDArray x2mag = x2.norm2(1);
		x1mag = Transforms.max(x1mag, Nd4j.EPS_THRESHOLD, false);
		x2mag = Transforms.max(x2mag, Nd4j.EPS_THRESHOLD, false);

		INDArray out = x1.mul(x2);
		out.diviColumnVector(x1mag);
		out.diviColumnVector(x2mag);

		return out;
	}

	@Override
	public Pair<Gradient, INDArray[]> doBackward(boolean tbptt) {
		if (!canDoBackward())
			throw new IllegalStateException("Cannot do backward pass: errors not set");

		INDArray x1 = inputs[0];
		INDArray x2 = inputs[1];

		INDArray x1L2norm = x1.norm2(1);
		INDArray x2L2norm = x2.norm2(1);

		INDArray x1L2normSq = x1L2norm.mul(x1L2norm);
		INDArray x1Dotx2L1norm = x2.mul(x1).sum(1);
		INDArray x1EpsilonNext = x2.mulColumnVector(x1L2normSq);
		x1EpsilonNext.subi(x1.mulColumnVector(x1Dotx2L1norm));

		INDArray x2L2normSq = x2L2norm.mul(x2L2norm);
		INDArray x2Dotx1L1norm = x1.mul(x2).sum(1);
		INDArray x2EpsilonNext = x1.mulColumnVector(x2L2normSq);
		x2EpsilonNext.subi(x2.mulColumnVector(x2Dotx1L1norm));

		x1L2norm = Transforms.max(x1L2norm, Nd4j.EPS_THRESHOLD, false);
		x2L2norm = Transforms.max(x2L2norm, Nd4j.EPS_THRESHOLD, false);
		x2L2normSq = Transforms.max(x2L2normSq, Nd4j.EPS_THRESHOLD, false);

		x1EpsilonNext.diviColumnVector(x2L2norm);
		x1EpsilonNext.diviColumnVector(x1L2norm.mul(x1L2normSq));

		x2EpsilonNext.diviColumnVector(x1L2norm);
		x2EpsilonNext.diviColumnVector(x2L2norm.mul(x2L2normSq));

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

	@Override
	public String toString() {
		return "CosineVertex(id=" + this.getVertexIndex() + ",name=\"" + this.getVertexName() + "\"" + ")";
	}
}
