package personal.pan.dl4j.examples.gan;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import javax.imageio.ImageIO;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.graph.PreprocessorVertex;
import org.deeplearning4j.nn.conf.graph.StackVertex;
import org.deeplearning4j.nn.conf.graph.UnstackVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ActivationLayer;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.Upsampling2D;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.impl.ActivationLReLU;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import personal.pan.dl4j.nn.conf.layers.NoParamOutputLayer;
import personal.pan.dl4j.nn.visual.MNISTVisualizer;

/**
 * Conditional Gan<br />
 * 待调优
 * 
 * @author Jerry
 *
 */
public class ConditionalConvGanTrainer {

	private final static String PREFIX = "D:\\soft\\test\\generator";

	static double lrD = 8e-4;
	static double lrG = lrD * 0.1;

	static DataType dataType = DataType.FLOAT;

	static IUpdater updaterD = new RmsProp(lrD);
	static IUpdater updaterG = new RmsProp(lrG);

	static int seed = 12345;
	static int epochs = 200000;

	static int height = 28;
	static int width = 28;
	static int channels = 1;
	static int batchSize = 60;
	static int vectorSize = 10;
	static int numClasses = 10;

	public static void main(String[] args) {
		train();
	}

	/**
	 * deconvolution_out = s(i−1)+k−2p<br>
	 * convolution_out = (i-k+2p)/s + 1
	 */
	public static void train() {

		ComputationGraph discriminator = null;

		try {
			File file = new File(PREFIX + "\\model\\Gan_original.zip");
			MnistDataSetIterator trainDataSetIterator = new MnistDataSetIterator(batchSize, true, seed);

			if (file.exists()) {
				discriminator = ComputationGraph.load(file, true);
			} else {
				ComputationGraphConfiguration graphConfiguration = new NeuralNetConfiguration.Builder().seed(seed)
						.dataType(dataType).weightInit(WeightInit.XAVIER).graphBuilder().addInputs("x", "y", "z")
						.setInputTypes(InputType.feedForward(height * width * channels),
								InputType.convolutional(height, width, numClasses),
								InputType.feedForward(vectorSize + numClasses))

						/* -------------------------Gz------------------------- */
						.addLayer("Gz_1", new DenseLayer.Builder().nIn(vectorSize + numClasses).nOut(16 * 16 * 4)
								.activation(Activation.RELU).weightInit(WeightInit.XAVIER).updater(updaterG).build(),
								"z")
						.addVertex("Gz_ffToCnn", new PreprocessorVertex(new FeedForwardToCnnPreProcessor(16, 16, 4)),
								"Gz_1")

						.addLayer("Gz_up_2", new Upsampling2D.Builder(2).build(), "Gz_ffToCnn")// width=32,height=32

						.addLayer("Gz_conv_2", new ConvolutionLayer.Builder(3, 3).updater(updaterG).nOut(64).build(),
								"Gz_up_2")// width=32,height=32 -> width=30,height=30

						.addLayer("Gz_bn_2", new BatchNormalization.Builder().decay(0.8).updater(updaterG).build(),
								"Gz_conv_2")

						.addLayer("Gz_activation_2", new ActivationLayer(Activation.RELU), "Gz_bn_2")

						.addLayer("Gz_final",
								new ConvolutionLayer.Builder(3, 3).updater(updaterG).activation(Activation.SIGMOID)
										.nOut(channels).build(),
								"Gz_activation_2")// width=30,height=30 -> width=28,height=28

						/* -------------------------Gz------------------------- */

						.addVertex("x_4d",
								new PreprocessorVertex(new FeedForwardToCnnPreProcessor(height, width, channels)), "x")
						.addVertex("merge_x", new MergeVertex(), "x_4d", "y")

						.addVertex("merge_Gz", new MergeVertex(), "Gz_final", "y")

						.addVertex("stack", new StackVertex(), "merge_x", "merge_Gz")

						/* -------------------------D------------------------- */

						.addLayer("D_conv_1",
								new ConvolutionLayer.Builder(5, 5).updater(updaterD).stride(1, 1)
										.nIn(channels + numClasses).nOut(20).build(),
								"stack")

						.addLayer("D_activation_1", new ActivationLayer(new ActivationLReLU(0.2)), "D_conv_1")

						.addLayer("D_conv_2",
								new ConvolutionLayer.Builder(5, 5).updater(updaterD).stride(1, 1).nIn(20).nOut(50)
										.build(),
								"D_activation_1")

						.addLayer("D_bn_2", new BatchNormalization.Builder().decay(0.8).updater(updaterD).build(),
								"D_conv_2")

						.addLayer("D_activation_2", new ActivationLayer(new ActivationLReLU(0.2)), "D_bn_2")

						.addLayer("D_final",
								new DenseLayer.Builder().nOut(1).updater(updaterD).activation(new ActivationLReLU(0.2))
										.build(),
								"D_activation_2")

						/* -------------------------D------------------------- */

						.addVertex("D(x)", new UnstackVertex(0, 2), "D_final")
						.addVertex("D(Gz)", new UnstackVertex(1, 2), "D_final")

						.addLayer("output_D(x)",
								new NoParamOutputLayer.Builder(LossFunction.XENT).updater(updaterD)
										.activation(Activation.SIGMOID).nOut(1).build(),
								"D(x)")
						.addLayer("output_D(Gz)",
								new NoParamOutputLayer.Builder(LossFunction.XENT).updater(updaterD)
										.activation(Activation.SIGMOID).nOut(1).build(),
								"D(Gz)")

						.setOutputs("output_D(x)", "output_D(Gz)").build();

				discriminator = new ComputationGraph(graphConfiguration);
				discriminator.init();
			}

			MNISTVisualizer bestVisualizer = new MNISTVisualizer(1, "ConditionalGan");

			MnistDataSetIterator testDataSetIterator = new MnistDataSetIterator(30, false, seed);

			int n = 0;
			for (int i = 0; i < epochs; i++) {
				while (trainDataSetIterator.hasNext()) {
					if (!testDataSetIterator.hasNext()) {
						testDataSetIterator.reset();
					}

					DataSet trainDataSet = trainDataSetIterator.next();

					INDArray inputX = trainDataSet.getFeatures().castTo(dataType);
					INDArray label = trainDataSet.getLabels().castTo(dataType);

					long num = inputX.size(0);

					INDArray inputZ = Nd4j.randn(dataType, new long[] { num, vectorSize });
					INDArray labelDx = Nd4j.ones(dataType, new long[] { num, 1 });
					INDArray labelDgz = Nd4j.zeros(dataType, new long[] { num, 1 });
					INDArray labelDgzT = Nd4j.ones(dataType, new long[] { num, 1 });

					INDArray label4d = label.reshape(num, label.columns(), 1, 1);
					INDArray ones4d = Nd4j.ones(dataType, new long[] { num, label.columns(), height, width });

					INDArray inputY = ones4d.muli(label4d);
					INDArray h = Nd4j.hstack(inputZ, label);

					MultiDataSet dataSetD = new org.nd4j.linalg.dataset.MultiDataSet(
							new INDArray[] { inputX, inputY, h }, new INDArray[] { labelDx, labelDgz });

					for (int k = 0; k < 1; k++) {
						discriminator.fit(dataSetD);
					}

					Map<String, INDArray> discriminatorActivations = discriminator.feedForward();
					System.out.println(discriminatorActivations.get("output_D(x)"));// 最后得平衡在0.5
					System.out.println(discriminatorActivations.get("output_D(Gz)"));// 最后得平衡在0.5
					System.out.println("-------------------------");

					if (n % 20 == 0) {
						DataSet testDataSet = testDataSetIterator.next();
						INDArray testX = testDataSet.getFeatures().castTo(dataType);
						INDArray testLabel = testDataSet.getLabels().castTo(dataType);

						long numTest = testX.size(0);

						INDArray label4dTest = testLabel.reshape(numTest, testLabel.columns(), 1, 1);
						INDArray ones4dTest = Nd4j.ones(dataType,
								new long[] { numTest, testLabel.columns(), height, width });

						INDArray testY = ones4dTest.muli(label4dTest);
						INDArray testZ = Nd4j.randn(dataType, new long[] { numTest, vectorSize });

						Map<String, INDArray> generatorActivations = discriminator
								.feedForward(new INDArray[] { testX, testY, Nd4j.hstack(testZ, testLabel) }, false);
						INDArray gz = generatorActivations.get("Gz_final").dup();

						List<INDArray> list = new ArrayList<INDArray>();
						for (int j = 0; j < gz.size(0); j++) {
							INDArray a = gz.get(new INDArrayIndex[] { NDArrayIndex.point(j), NDArrayIndex.all(),
									NDArrayIndex.all(), NDArrayIndex.all() });
							list.add(a);
						}

						bestVisualizer.setDigits(list);
						bestVisualizer.visualize();

						writeImage(PREFIX + "\\aaaa_" + n + ".jpg", gz);
					}

					saveModel(discriminator, n);

					frozen(discriminator, true);

					MultiDataSet dataSetG = new org.nd4j.linalg.dataset.MultiDataSet(
							new INDArray[] { inputX, inputY, h }, new INDArray[] { labelDx, labelDgzT });

					for (int k = 0; k < 20; k++) {
						discriminator.fit(dataSetG);
					}

					frozen(discriminator, false);

					n++;
				}

				trainDataSetIterator.reset();
				System.out.println("reset");
			}
		} catch (Exception | Error e) {
			e.printStackTrace();
		}
	}

	public static void test() {
		try {
			File file = new File(PREFIX + "\\model\\Gan_3060.zip");
			ComputationGraph discriminator = ComputationGraph.load(file, true);

			MnistDataSetIterator testDataSetIterator = new MnistDataSetIterator(30, false, seed);

			MNISTVisualizer bestVisualizer = new MNISTVisualizer(1, "ConditionalGan");

			while (testDataSetIterator.hasNext()) {
				DataSet testDataSet = testDataSetIterator.next();
				INDArray testX = testDataSet.getFeatures().castTo(dataType);
				INDArray testLabel = testDataSet.getLabels().castTo(dataType);

				long numTest = testX.size(0);

				INDArray label4dTest = testLabel.reshape(numTest, testLabel.columns(), 1, 1);
				INDArray ones4dTest = Nd4j.ones(dataType, new long[] { numTest, testLabel.columns(), height, width });

				INDArray testY = ones4dTest.muli(label4dTest);
				INDArray testZ = Nd4j.randn(dataType, new long[] { numTest, vectorSize });

				Map<String, INDArray> generatorActivations = discriminator
						.feedForward(new INDArray[] { testX, testY, Nd4j.hstack(testZ, testLabel) }, false);
				INDArray gz = generatorActivations.get("Gz_final").dup();

				List<INDArray> list = new ArrayList<INDArray>();
				for (int j = 0; j < gz.size(0); j++) {
					INDArray a = gz.get(new INDArrayIndex[] { NDArrayIndex.point(j), NDArrayIndex.all(),
							NDArrayIndex.all(), NDArrayIndex.all() });
					list.add(a);
				}

				bestVisualizer.setDigits(list);
				bestVisualizer.visualize();

				Thread.sleep(3000L);
			}

			testDataSetIterator.reset();
		} catch (Exception | Error e) {
			e.printStackTrace();
		}
	}

	@SuppressWarnings("rawtypes")
	static void frozen(ComputationGraph discriminator, boolean flag) {
		Layer[] layers = discriminator.getLayers();
		for (Layer layer : layers) {
			if (layer instanceof BaseLayer) {
				BaseLayer baseLayer = (BaseLayer) layer;
				org.deeplearning4j.nn.conf.layers.Layer l = baseLayer.getConf().getLayer();
				org.deeplearning4j.nn.conf.layers.BaseLayer bl = (org.deeplearning4j.nn.conf.layers.BaseLayer) l;

				IUpdater u = bl.getIUpdater();
				String layerName = bl.getLayerName();
				if (flag) {
					if (layerName.startsWith("Gz_")) {
						u.setLrAndSchedule(lrG, null);
					} else if (layerName.startsWith("D_")) {
						u.setLrAndSchedule(0, null);
					}
				} else {
					if (layerName.startsWith("Gz_")) {
						u.setLrAndSchedule(0, null);
					} else if (layerName.startsWith("D_")) {
						u.setLrAndSchedule(lrD, null);
					}
				}
			}
		}
	}

	static void writeImage(String path, INDArray indArray) {
		try {
			BufferedImage bufferedImage = imageFromINDArray(indArray);
			if (bufferedImage == null) {
				System.out.println("(writeImage) bufferedImage == null");
			}
			ImageIO.write(bufferedImage, "jpg", new File(path));
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private static BufferedImage imageFromINDArray(INDArray array) {
		BufferedImage image = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
		for (int i = 0; i < 784; i++) {
			image.getRaster().setSample(i % 28, i / 28, 0, (int) (255 * array.getDouble(i)));
		}

		return image;
	}

	static void saveModel(ComputationGraph discriminator, int n) throws Exception {
		if (n % 2 == 0) {
			discriminator.save(new File(PREFIX + "\\model\\Gan_" + n + ".zip"));
		}
	}

}
