package personal.pan.dl4j.nn.graph.rnn;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.BaseGraphVertex;
import org.deeplearning4j.nn.graph.vertex.VertexIndices;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.primitives.Pair;

public class AverageTimeStepVertex extends BaseGraphVertex {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private String inputName;
	private int inputIdx;

	public AverageTimeStepVertex(ComputationGraph graph, String name, int vertexIndex, String inputName) {
		this(graph, name, vertexIndex, null, null, inputName);
	}

	public AverageTimeStepVertex(ComputationGraph graph, String name, int vertexIndex, VertexIndices[] inputVertices,
			VertexIndices[] outputVertices, String inputName) {
		super(graph, name, vertexIndex, inputVertices, outputVertices);
		this.inputName = inputName;
		this.inputIdx = graph.getConfiguration().getNetworkInputs().indexOf(inputName);
		if (inputIdx == -1)
			throw new IllegalArgumentException("Invalid input name: \"" + inputName + "\" not found in list "
					+ "of network inputs (" + graph.getConfiguration().getNetworkInputs() + ")");
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
		INDArray out = inputs[0].sum(2);
		return out;
	}

	@Override
	public Pair<Gradient, INDArray[]> doBackward(boolean tbptt) {
		INDArray epsilonsOut = Nd4j.create(inputs[0].shape());

		int lastTS = inputs[0].size(2) - 1;
		INDArray average = epsilon;
		for (int i = 0; i <= lastTS; i++) {
			epsilonsOut.put(new INDArrayIndex[] { NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(i) },
					average);
		}

		return new Pair<>(null, new INDArray[] { epsilonsOut });
	}

	@Override
	public void setBackpropGradientsViewArray(INDArray backpropGradientsViewArray) {
		if (backpropGradientsViewArray != null)
			throw new RuntimeException("Vertex does not have gradients; gradients view array cannot be set here");
	}

	@Override
	public Pair<INDArray, MaskState> feedForwardMaskArrays(INDArray[] maskArrays, MaskState currentMaskState,
			int minibatchSize) {
		return new Pair<>(null, currentMaskState);
	}

	@Override
	public String toString() {
		return "AverageTimeStepVertex(inputName=" + inputName + ")";
	}
}
