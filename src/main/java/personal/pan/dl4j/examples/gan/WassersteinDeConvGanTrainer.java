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
import org.deeplearning4j.nn.conf.graph.PreprocessorVertex;
import org.deeplearning4j.nn.conf.graph.StackVertex;
import org.deeplearning4j.nn.conf.graph.UnstackVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ActivationLayer;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.Deconvolution2D;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LossLayer;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
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
import org.nd4j.linalg.lossfunctions.impl.LossWasserstein;

import personal.pan.dl4j.nn.visual.MNISTVisualizer;

/**
 * 效果欠佳 优化中<br />
 * <br />
 * Discriminator使用Cnn，输出层损失函数使用{@link LossWasserstein} <br />
 * Generator使用转置Cnn，最后一层激活函数使用SIGMOID<br />
 * <br />
 * D(x)和D(Gz)的差值需要接近0<br />
 * 本例生成的图片单一
 * 
 * @author Gerry
 *
 */
public class WassersteinDeConvGanTrainer {

	private final static String PREFIX = "D:\\soft\\test\\generator";

	static double lrD = 1e-4;
	static double lrG = lrD * 0.5;

	static DataType dataType = DataType.FLOAT;

	static IUpdater updaterD = new RmsProp(lrD);
	static IUpdater updaterG = new RmsProp(lrG);

	static int seed = 12345;
	static int epochs = 200000;

	static int height = 28;
	static int width = 28;
	static int channels = 1;
	static int batchSize = 200;
	static int vectorSize = 10;
	static double gt = 0.01;

	private WassersteinDeConvGanTrainer() {
	}

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
						.dataType(dataType).weightInit(WeightInit.XAVIER).graphBuilder().addInputs("x", "z")
						.setInputTypes(InputType.feedForward(height * width * channels),
								InputType.feedForward(vectorSize))

						/* -------------------------Gz------------------------- */
						.addLayer("Gz_liner",
								new DenseLayer.Builder().nIn(vectorSize).nOut(4 * 4 * 256).activation(Activation.RELU)
										.updater(updaterG).build(),
								"z")
						.addVertex("Gz_ffToCnn", new PreprocessorVertex(new FeedForwardToCnnPreProcessor(4, 4, 256)),
								"Gz_liner")
						.addLayer("Gz_deconv_1",
								new Deconvolution2D.Builder(4, 4).updater(updaterG).stride(1, 1).nIn(256).nOut(128)
										.build(),
								"Gz_ffToCnn")
						.addLayer("Gz_bn_1",
								new BatchNormalization.Builder().updater(updaterG).nIn(128).nOut(128).decay(0.8)
										.build(),
								"Gz_deconv_1")
						.addLayer("Gz_activation_1", new ActivationLayer(Activation.RELU), "Gz_bn_1")
						.addLayer("Gz_deconv_2",
								new Deconvolution2D.Builder(2, 2).updater(updaterG).stride(2, 2).nIn(128).nOut(64)
										.build(),
								"Gz_activation_1")
						.addLayer("Gz_bn_2",
								new BatchNormalization.Builder().updater(updaterG).nIn(64).nOut(64).decay(0.8).build(),
								"Gz_deconv_2")
						.addLayer("Gz_activation_2", new ActivationLayer(Activation.RELU), "Gz_bn_2")
						.addLayer("Gz_deconv_3",
								new Deconvolution2D.Builder(2, 2).updater(updaterG).stride(2, 2).nIn(64).nOut(channels)
										.activation(Activation.SIGMOID).build(),
								"Gz_activation_2")
						.addVertex("Gz_final",
								new PreprocessorVertex(new CnnToFeedForwardPreProcessor(height, width, channels)),
								"Gz_deconv_3")

						/* -------------------------Gz------------------------- */

						.addVertex("stack", new StackVertex(), "x", "Gz_final")

						/* -------------------------D------------------------- */
						.addVertex("D_cnnToFf",
								new PreprocessorVertex(new FeedForwardToCnnPreProcessor(height, width, channels)),
								"stack")

						.addLayer("D_conv_1",
								new ConvolutionLayer.Builder(5, 5).updater(updaterD).stride(1, 1).nIn(channels).nOut(20)
										.gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
										.gradientNormalizationThreshold(gt).build(),
								"D_cnnToFf")

						.addLayer("D_activation_1", new ActivationLayer(new ActivationLReLU(0.2)), "D_conv_1")

						.addLayer("D_conv_2",
								new ConvolutionLayer.Builder(5, 5).updater(updaterD).stride(1, 1).nIn(20).nOut(50)
										.gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
										.gradientNormalizationThreshold(gt).build(),
								"D_activation_1")

						.addLayer("D_activation_2", new ActivationLayer(new ActivationLReLU(0.2)), "D_conv_2")

						.addLayer("D_final",
								new DenseLayer.Builder().nOut(1).updater(updaterD).activation(Activation.IDENTITY)
										.gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
										.gradientNormalizationThreshold(gt).build(),
								"D_activation_2")

						/* -------------------------D------------------------- */

						.addVertex("D(x)", new UnstackVertex(0, 2), "D_final")
						.addVertex("D(Gz)", new UnstackVertex(1, 2), "D_final")

						.addLayer("output_D(x)", new LossLayer.Builder(LossFunction.WASSERSTEIN).build(), "D(x)")
						.addLayer("output_D(Gz)", new LossLayer.Builder(LossFunction.WASSERSTEIN).build(), "D(Gz)")

						.setOutputs("output_D(x)", "output_D(Gz)").build();

				discriminator = new ComputationGraph(graphConfiguration);
				discriminator.init();
			}

			MNISTVisualizer bestVisualizer = new MNISTVisualizer(1, "WassersteinGan");

			MnistDataSetIterator testDataSetIterator = new MnistDataSetIterator(30, false, seed);

			int n = 0;
			for (int i = 0; i < epochs; i++) {
				while (trainDataSetIterator.hasNext()) {
					if (!testDataSetIterator.hasNext()) {
						testDataSetIterator.reset();
					}

					INDArray inputX = trainDataSetIterator.next().getFeatures().castTo(dataType);
					long num = inputX.size(0);

					INDArray inputZ = Nd4j.randn(dataType, new long[] { num, vectorSize });
					INDArray labelDx = Nd4j.ones(dataType, new long[] { num, 1 });
					INDArray labelDgz = Nd4j.ones(dataType, new long[] { num, 1 }).mul(-1);// LossWasserstein建议使用-1
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

					if (n % 10 == 0) {
						DataSet testDataSet = testDataSetIterator.next();
						INDArray testX = testDataSet.getFeatures().castTo(dataType);
						INDArray z = Nd4j.randn(dataType, new long[] { testX.size(0), vectorSize });

						Map<String, INDArray> generatorActivations = discriminator
								.feedForward(new INDArray[] { testX, z }, false);
						INDArray gz = generatorActivations.get("Gz_final").dup();

						List<INDArray> list = new ArrayList<INDArray>();
						for (int j = 0; j < gz.size(0); j++) {
							INDArray a = gz.get(new INDArrayIndex[] { NDArrayIndex.point(j), NDArrayIndex.all() });
							list.add(a);
						}

						bestVisualizer.setDigits(list);
						bestVisualizer.visualize();

						writeImage(PREFIX + "\\aaaa_" + n + ".jpg", gz);
					}

					saveModel(discriminator, n);

					frozen(discriminator, true);

					MultiDataSet dataSetG = new org.nd4j.linalg.dataset.MultiDataSet(new INDArray[] { inputX, inputZ },
							new INDArray[] { labelDx, labelDgzT });

					for (int k = 0; k < 30; k++) {
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

	static void draw(ComputationGraph discriminator, MnistDataSetIterator testDataSetIterator,
			MNISTVisualizer bestVisualizer, int i) throws Exception {
		if (i % 10 == 0) {
			DataSet testDataSet = testDataSetIterator.next();
			INDArray testX = testDataSet.getFeatures().castTo(dataType);
			INDArray z = Nd4j.randn(dataType, new long[] { testX.size(0), vectorSize });

			Map<String, INDArray> generatorActivations = discriminator.feedForward(new INDArray[] { testX, z }, false);
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
	}

	static void saveModel(ComputationGraph discriminator, int i) throws Exception {
		if (i % 5 == 0) {
			discriminator.save(new File(PREFIX + "\\model\\Gan_" + i + ".zip"));
		}
	}

}
