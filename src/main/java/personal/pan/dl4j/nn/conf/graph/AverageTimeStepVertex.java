package personal.pan.dl4j.nn.conf.graph;

import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.jackson.annotation.JsonProperty;

public class AverageTimeStepVertex extends GraphVertex {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private String maskArrayInputName;

	public AverageTimeStepVertex() {

	}

	/**
	 *
	 * @param maskArrayInputName
	 *            The name of the input to look at when determining the last time
	 *            step. Specifically, the mask array of this time series input is
	 *            used when determining which time step to extract and return.
	 */
	public AverageTimeStepVertex(@JsonProperty("maskArrayInputName") String maskArrayInputName) {
		this.maskArrayInputName = maskArrayInputName;
	}

	@Override
	public GraphVertex clone() {
		return new LastTimeStepVertex(maskArrayInputName);
	}

	@Override
	public boolean equals(Object o) {
		if (!(o instanceof LastTimeStepVertex))
			return false;
		LastTimeStepVertex ltsv = (LastTimeStepVertex) o;
		if (maskArrayInputName == null && ltsv.getMaskArrayInputName() != null
				|| maskArrayInputName != null && ltsv.getMaskArrayInputName() == null)
			return false;
		return maskArrayInputName == null || maskArrayInputName.equals(ltsv.getMaskArrayInputName());
	}

	@Override
	public int hashCode() {
		return (maskArrayInputName == null ? 452766971 : maskArrayInputName.hashCode());
	}

	@Override
	public int numParams(boolean backprop) {
		return 0;
	}

	@Override
	public int minVertexInputs() {
		return 1;
	}

	@Override
	public int maxVertexInputs() {
		return 1;
	}

	@Override
	public personal.pan.dl4j.nn.graph.rnn.AverageTimeStepVertex instantiate(ComputationGraph graph, String name,
			int idx, INDArray paramsView, boolean initializeParams) {
		return new personal.pan.dl4j.nn.graph.rnn.AverageTimeStepVertex(graph, name, idx, maskArrayInputName);
	}

	@Override
	public InputType getOutputType(int layerIndex, InputType... vertexInputs) throws InvalidInputTypeException {
		if (vertexInputs.length != 1)
			throw new InvalidInputTypeException("Invalid input type: cannot get last time step of more than 1 input");
		if (vertexInputs[0].getType() != InputType.Type.RNN) {
			throw new InvalidInputTypeException(
					"Invalid input type: cannot get subset of non RNN input (got: " + vertexInputs[0] + ")");
		}

		return InputType.feedForward(((InputType.InputTypeRecurrent) vertexInputs[0]).getSize());
	}

	@Override
	public MemoryReport getMemoryReport(InputType... inputTypes) {
		// No additional working memory (beyond activations/epsilons)
		return new LayerMemoryReport.Builder(null, LastTimeStepVertex.class, inputTypes[0],
				getOutputType(-1, inputTypes)).standardMemory(0, 0).workingMemory(0, 0, 0, 0).cacheMemory(0, 0).build();
	}

	public String getMaskArrayInputName() {
		return maskArrayInputName;
	}

	public void setMaskArrayInputName(String maskArrayInputName) {
		this.maskArrayInputName = maskArrayInputName;
	}

	@Override
	public String toString() {
		return "LastTimeStepVertex(inputName=" + maskArrayInputName + ")";
	}
}
