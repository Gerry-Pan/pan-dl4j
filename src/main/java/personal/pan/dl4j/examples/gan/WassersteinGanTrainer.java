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
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.StackVertex;
import org.deeplearning4j.nn.conf.graph.UnstackVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.DropoutLayer;
import org.deeplearning4j.nn.conf.layers.LossLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.impl.ActivationLReLU;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import personal.pan.dl4j.nn.visual.MNISTVisualizer;

public class WassersteinGanTrainer {

	private final static String PREFIX = "D:\\soft\\test\\generator";

	static double lrD = 1e-4;
	static double lrG = lrD * 0.1;

	static DataType dataType = DataType.FLOAT;

	static IUpdater updaterD = new RmsProp(lrD);
	static IUpdater updaterG = new RmsProp(lrG);

	static int seed = 12345;
	static int epochs = 200000;

	static int height = 28;
	static int width = 28;
	static int channels = 1;
	static int batchSize = 200;
	static int vectorSize = 20;
	static double gt = 1e-3;

	private WassersteinGanTrainer() {
	}

	public static void main(String[] args) {
		train();
	}

	/**
	 * 
	 */
	public static void train() {

		ComputationGraph discriminator = null;

		try {
			MnistDataSetIterator trainDataSetIterator = new MnistDataSetIterator(batchSize, true, seed);

			ComputationGraphConfiguration discriminatorConfig = new NeuralNetConfiguration.Builder().seed(seed)
					.dataType(dataType).weightInit(WeightInit.XAVIER).graphBuilder().addInputs("x", "z")
					.setInputTypes(InputType.feedForward(height * width * channels), InputType.feedForward(vectorSize))

					/* -------------------------Gz------------------------- */
					.addLayer("Gz_1",
							new DenseLayer.Builder().nIn(vectorSize).nOut(512).activation(Activation.RELU)
									.weightInit(WeightInit.XAVIER).updater(updaterG).build(),
							"z")
					.addLayer("Gz_2",
							new DenseLayer.Builder().nIn(512).nOut(256).activation(Activation.RELU)
									.weightInit(WeightInit.XAVIER).updater(updaterG).build(),
							"Gz_1")
					.addLayer("Gz_final",
							new DenseLayer.Builder().nIn(256).nOut(height * width * channels).updater(updaterG)
									.activation(Activation.SIGMOID).weightInit(WeightInit.XAVIER).build(),
							"Gz_2")
					/* -------------------------Gz------------------------- */

					.addVertex("stack", new StackVertex(), "x", "Gz_final")

					/* -------------------------D------------------------- */
					.addLayer("D_1", new DenseLayer.Builder().nIn(height * width * channels).nOut(256)
							.activation(new ActivationLReLU(0.2)).weightInit(WeightInit.XAVIER).updater(updaterD)
							.gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
							.gradientNormalizationThreshold(gt).build(), "stack")

					.addLayer("D_dropout_1", new DropoutLayer(0.5), "D_1")

					.addLayer("D_2",
							new DenseLayer.Builder().nIn(256).nOut(128).activation(new ActivationLReLU(0.2))
									.weightInit(WeightInit.XAVIER).updater(updaterD)
									.gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
									.gradientNormalizationThreshold(gt).build(),
							"D_dropout_1")

					.addLayer("D_dropout_2", new DropoutLayer(0.5), "D_2")

					.addLayer("D_final",
							new DenseLayer.Builder().nIn(128).nOut(1).activation(Activation.IDENTITY)
									.weightInit(WeightInit.XAVIER).updater(updaterD)
									.gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
									.gradientNormalizationThreshold(gt).build(),
							"D_dropout_2")
					/* -------------------------D------------------------- */

					.addVertex("D(x)", new UnstackVertex(0, 2), "D_final")
					.addVertex("D(Gz)", new UnstackVertex(1, 2), "D_final")

					.addLayer("output_D(x)", new LossLayer.Builder(LossFunction.WASSERSTEIN).build(), "D(x)")
					.addLayer("output_D(Gz)", new LossLayer.Builder(LossFunction.WASSERSTEIN).build(), "D(Gz)")

					.setOutputs("output_D(x)", "output_D(Gz)").build();

			discriminator = new ComputationGraph(discriminatorConfig);
			discriminator.init();

			boolean flag = false;
			MNISTVisualizer bestVisualizer = new MNISTVisualizer(1, "WassersteinGan");

			MnistDataSetIterator testDataSetIterator = new MnistDataSetIterator(30, false, seed);

			for (int i = 0; i < epochs; i++) {
				if (!trainDataSetIterator.hasNext()) {
					trainDataSetIterator.reset();
				}
				if (!testDataSetIterator.hasNext()) {
					testDataSetIterator.reset();
				}

				INDArray inputX = trainDataSetIterator.next().getFeatures().castTo(dataType);
				long num = inputX.size(0);

				INDArray inputZ = Nd4j.randn(dataType, new long[] { num, vectorSize });
				INDArray labelDx = Nd4j.ones(dataType, new long[] { num, 1 });
				INDArray labelDgz = Nd4j.zeros(dataType, new long[] { num, 1 });
				INDArray labelDgzT = Nd4j.ones(dataType, new long[] { num, 1 });

				MultiDataSet dataSetD = new org.nd4j.linalg.dataset.MultiDataSet(new INDArray[] { inputX, inputZ },
						new INDArray[] { labelDx, labelDgz });

				for (int k = 0; k < 1; k++) {
					discriminator.fit(dataSetD);
				}

				Map<String, INDArray> discriminatorActivations = discriminator.feedForward();
				INDArray D_x = discriminatorActivations.get("output_D(x)");
				INDArray D_Gz = discriminatorActivations.get("output_D(Gz)");

				System.out.println(D_x.sub(D_Gz));// 需要趋于0
				System.out.println("-------------------------");

				if (i % 50 == 0) {
					INDArray testX = testDataSetIterator.next().getFeatures().castTo(dataType);
					INDArray z = Nd4j.randn(dataType, new long[] { testX.size(0), vectorSize });

					Map<String, INDArray> generatorActivations = discriminator.feedForward(new INDArray[] { testX, z },
							false);
					INDArray gz = generatorActivations.get("Gz_final").dup();

					List<INDArray> list = new ArrayList<INDArray>();
					for (int j = 0; j < gz.size(0); j++) {
						INDArray a = gz.get(new INDArrayIndex[] { NDArrayIndex.point(j), NDArrayIndex.all() });
						list.add(a);
					}

					bestVisualizer.setDigits(list);
					bestVisualizer.visualize();

					writeImage(PREFIX + "\\aaaa_" + i + ".jpg", gz);
					saveModel(discriminator, i);
				}

				flag = true;

				frozen(discriminator, flag);

				MultiDataSet dataSetG = new org.nd4j.linalg.dataset.MultiDataSet(new INDArray[] { inputX, inputZ },
						new INDArray[] { labelDx, labelDgzT });

				for (int k = 0; k < 30; k++) {
					discriminator.fit(dataSetG);
				}

				flag = false;

				frozen(discriminator, flag);
			}

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

	static void saveModel(ComputationGraph discriminator, int i) throws Exception {
//		discriminator.save(new File(PREFIX + "\\model\\Gan_" + i + ".zip"));
	}

}
