package personal.pan.dl4j.nn.conf.graph;

import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * 
 * @author Jerry
 *
 */
@Deprecated
public class AttentionVertex extends GraphVertex {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	@Override
	public GraphVertex clone() {
		return new AttentionVertex();
	}

	@Override
	public boolean equals(Object o) {
		return this.equals(o);
	}

	@Override
	public int hashCode() {
		return this.hashCode();
	}

	@Override
	public long numParams(boolean backprop) {
		return 0;
	}

	@Override
	public int minVertexInputs() {
		return 2;
	}

	@Override
	public int maxVertexInputs() {
		return 2;
	}

	@Override
	public org.deeplearning4j.nn.graph.vertex.GraphVertex instantiate(ComputationGraph graph, String name, int idx,
			INDArray paramsView, boolean initializeParams, DataType networkDatatype) {
		return new personal.pan.dl4j.nn.graph.AttentionVertex(graph, name, idx, networkDatatype);
	}

	@Override
	public InputType getOutputType(int layerIndex, InputType... vertexInputs) throws InvalidInputTypeException {
		return vertexInputs[0];
	}

	@Override
	public MemoryReport getMemoryReport(InputType... inputTypes) {
		return null;
	}

}
