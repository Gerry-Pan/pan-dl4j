package com.pan.test;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import javax.imageio.ImageIO;

import org.datavec.image.loader.Java2DNativeImageLoader;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
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
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import personal.pan.dl4j.nn.conf.layers.NoParamOutputLayer;

public class Gan2Test {

	protected static double lr = 0.01;
	protected static double LowerBound = 0;

	protected static DataType dataType = DataType.FLOAT;

	protected static IUpdater frozenUpdater = new Adam(0.00);
	protected static IUpdater updaterD = new RmsProp(lr);
	protected static IUpdater updaterG = new RmsProp(lr);

	public static void main(String[] args) {
		train(args);
	}

	/**
	 * deconvolution_out = s(i−1)+k−2p<br>
	 * convolution_out = (i-k+2p)/s + 1
	 */
	public static void train(String[] args) {
		int seed = 12345;
		int epochs = 2000;

		int height = 28;
		int width = 28;
		int channels = 1;
		int batchSize = 5;
		int vectorSize = height * width * channels;
		ComputationGraph discriminator = null;
		Java2DNativeImageLoader imageLoader = new Java2DNativeImageLoader(height, width, channels);

		try {
			List<File> dataList = load("D:\\test\\image");

			if (dataList == null || dataList.isEmpty()) {
				return;
			}

			GanMultiDataSetIterator dataSetIterator = new GanMultiDataSetIterator(batchSize, height, width, channels,
					vectorSize, dataList, dataType);

			ComputationGraphConfiguration discriminatorConfig = new NeuralNetConfiguration.Builder().seed(seed)
					.dataType(dataType).weightInit(WeightInit.XAVIER).graphBuilder().addInputs("x", "z")
					.setInputTypes(InputType.convolutional(height, width, channels),
							InputType.convolutional(height, width, channels))

					/* -------------------------Gz------------------------- */
					.addLayer("Gz_liner",
							new DenseLayer.Builder().nIn(vectorSize).nOut(4 * 4 * 512).activation(Activation.RELU)
									.updater(updaterG).build(),
							"z")
					.addVertex("Gz_ffToCnn", new PreprocessorVertex(new FeedForwardToCnnPreProcessor(4, 4, 512)),
							"Gz_liner")
					.addLayer("Gz_deconv_1",
							new Deconvolution2D.Builder(4, 4).updater(updaterG).stride(1, 1).nIn(512).nOut(256).build(),
							"Gz_ffToCnn")
					.addLayer("Gz_bn_1",
							new BatchNormalization.Builder().updater(updaterG).nIn(256).nOut(256).decay(1).build(),
							"Gz_deconv_1")
					.addLayer("Gz_activation_1", new ActivationLayer(Activation.RELU), "Gz_bn_1")
					.addLayer("Gz_deconv_2",
							new Deconvolution2D.Builder(2, 2).updater(updaterG).stride(2, 2).nIn(256).nOut(128).build(),
							"Gz_activation_1")
					.addLayer("Gz_bn_2",
							new BatchNormalization.Builder().updater(updaterG).nIn(128).nOut(128).decay(1).build(),
							"Gz_deconv_2")
					.addLayer("Gz_activation_2", new ActivationLayer(Activation.RELU), "Gz_bn_2")
					.addLayer("Gz_final",
							new Deconvolution2D.Builder(2, 2).updater(updaterG).stride(2, 2).nIn(128).nOut(channels)
									.activation(Activation.RELU).build(),
							"Gz_activation_2")
					/* -------------------------Gz------------------------- */

					.addVertex("stack", new StackVertex(), "x", "Gz_final")

					/* -------------------------D------------------------- */
					.addLayer("D_conv_1",
							new ConvolutionLayer.Builder(5, 5).updater(updaterD).stride(1, 1).nIn(channels).nOut(20)
									.build(),
							"stack")

					.addLayer("D_subsample_1",
							new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2)
									.build(),
							"D_conv_1")

					.addLayer("D_bn_1", new BatchNormalization.Builder().updater(updaterD).build(), "D_subsample_1")

					.addLayer("D_activation_1", new ActivationLayer(Activation.RELU), "D_bn_1")

					.addLayer("D_conv_2",
							new ConvolutionLayer.Builder(5, 5).updater(updaterD).stride(1, 1).nIn(20).nOut(50).build(),
							"D_activation_1")

					.addLayer("D_subsample_2",
							new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2)
									.build(),
							"D_conv_2")

					.addLayer("D_bn_2", new BatchNormalization.Builder().updater(updaterD).build(), "D_subsample_2")

					.addLayer("D_activation", new ActivationLayer(Activation.RELU), "D_bn_2")

					.addLayer("D_final",
							new DenseLayer.Builder().nIn(4 * 4 * 50).nOut(2).updater(updaterD)
									.activation(Activation.RELU).build(),
							"D_activation")

					/* -------------------------D------------------------- */

					.addVertex("D(x)", new UnstackVertex(0, 2), "D_final")
					.addVertex("D(Gz)", new UnstackVertex(1, 2), "D_final")

					.addLayer("output_D(x)",
							new NoParamOutputLayer.Builder(LossFunction.MCXENT).activation(Activation.SOFTMAX).nOut(2)
									.build(),
							"D(x)")
					.addLayer("output_D(Gz)",
							new NoParamOutputLayer.Builder(LossFunction.MCXENT).activation(Activation.SOFTMAX).nOut(2)
									.build(),
							"D(Gz)")

					.setOutputs("output_D(x)", "output_D(Gz)").build();

			discriminator = new ComputationGraph(discriminatorConfig);
			discriminator.init();

			boolean flag = false;

			for (int i = 0; i < epochs; i++) {
				while (dataSetIterator.hasNext()) {
					MultiDataSet mds = dataSetIterator.next();
					discriminator.fit(mds);
				}

				Map<String, INDArray> discriminatorActivations = discriminator.feedForward();
				System.out.println(discriminatorActivations.get("output_D(x)"));
				System.out.println(discriminatorActivations.get("output_D(Gz)"));
				System.out.println("-------------------------");

				INDArray x = imageLoader.asMatrix(new File("E:\\soft\\new_minist\\数据\\test_data\\2.jpg"))
						.castTo(dataType);
				INDArray z = Nd4j.randn(dataType, new long[] { 1, vectorSize });
				Map<String, INDArray> generatorActivations = discriminator.feedForward(new INDArray[] { x, z }, false);
				INDArray gz = generatorActivations.get("Gz_final");

				System.out.println(gz.shape()[0]);
				writeImage("D:\\test\\generator\\image\\aaaa_" + i + ".jpg", gz);

				flag = true;
				dataSetIterator.reset();
				dataSetIterator.setFlag(flag);

				frozen(discriminator, flag);

				for (int k = 0; k < 5; k++) {
					while (dataSetIterator.hasNext()) {
						MultiDataSet mds = dataSetIterator.next();
						discriminator.fit(mds);
					}

					dataSetIterator.reset();
				}

				flag = false;
				dataSetIterator.reset();
				dataSetIterator.setFlag(flag);

				frozen(discriminator, flag);
			}

		} catch (Exception | Error e) {
			e.printStackTrace();
		}
	}

	@SuppressWarnings("rawtypes")
	protected static void frozen(ComputationGraph discriminator, boolean flag) {
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
						u.setLrAndSchedule(lr, null);
					} else if (layerName.startsWith("D_")) {
						u.setLrAndSchedule(0, null);
					}
				} else {
					if (layerName.startsWith("Gz_")) {
						u.setLrAndSchedule(0, null);
					} else if (layerName.startsWith("D_")) {
						u.setLrAndSchedule(lr, null);
					}
				}
			}
		}
	}

	protected static void createMatrixImage(String fileDir, int[][] matrix) throws IOException {
		int cx = matrix.length;
		int cy = matrix[0].length;
		// 填充矩形高宽
		int cz = 1;
		// 生成图的宽度
		int width = cx * cz;
		// 生成图的高度
		int height = cy * cz;

		BufferedImage bufferedImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
		Graphics2D gs = bufferedImage.createGraphics();
		gs.setBackground(Color.BLACK);
		gs.clearRect(0, 0, width, height);

		gs.setColor(Color.WHITE);
		for (int i = 0; i < cx; i++) {
			for (int j = 0; j < cy; j++) {
				// 1绘制填充黑矩形
				if (matrix[j][i] > 0) {
					gs.drawRect(i * cz, j * cz, cz, cz);
					gs.fillRect(i * cz, j * cz, cz, cz);
				}
			}
		}
		bufferedImage.flush();
		// 输出文件
		OutputStream output = new FileOutputStream(new File(fileDir));
		ImageIO.write(bufferedImage, "jpeg", output);
	}

	protected static int colorToRGB(int alpha, int red, int green, int blue) {
		int newPixel = 0;
		newPixel += alpha;
		newPixel = newPixel << 8;
		newPixel += red;
		newPixel = newPixel << 8;
		newPixel += green;
		newPixel = newPixel << 8;
		newPixel += blue;

		return newPixel;
	}

	protected static List<File> load(String path) {
		List<File> dataList = null;
		try {
			File root = new File(path);

			dataList = Arrays.asList(root.listFiles());

			Collections.shuffle(dataList);
		} catch (Exception e) {
			e.printStackTrace();

			throw new RuntimeException("加载数据失败...");
		} finally {

		}

		return dataList;
	}

	protected static INDArray readImage(File image, int width, int height, int channel, int demension) {
		INDArray indArray = null;

		if (image.exists()) {
			try {
				// [channel, width * height]
				BufferedImage bufferedImage = ImageIO.read(image);
				if (bufferedImage == null) {
					System.out.println("bufferedImage == null. path = " + image.getAbsolutePath());
				}
				double[] values = new double[channel * width * height];
				for (int i = 0; i < width; i++) {
					for (int j = 0; j < height; j++) {
						int r = 255, g = 255, b = 255;
						if (bufferedImage != null) {
							int rgb = bufferedImage.getRGB(j, i);
							Color color = new Color(rgb);
							// [0, 255]
							r = color.getRed();
							g = color.getGreen();
							b = color.getBlue();
						} else {
							System.out.println("bufferedImage is null");
						}
						// 将值转为[-1,1]范围
						double rValue = norm(r);

						values[(i * height + j)] = rValue;

						if (channel == 3) {
							double gValue = norm(g);
							double bValue = norm(b);

							values[(i * height + j) + 1 * width * height] = gValue;
							values[(i * height + j) + 2 * width * height] = bValue;
						}
					}
				}

				if (demension == 4) {
					indArray = Nd4j.create(values, new int[] { 1, channel, width, height }).castTo(dataType);
				} else if (demension == 2) {
					indArray = Nd4j.create(values, new int[] { 1, channel * width * height }).castTo(dataType);
				}
			} catch (IOException e) {

			}
		}

		return indArray;
	}

	protected static void writeImage(String path, INDArray indArray) {
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
		BufferedImage image = null;

		long[] shape = array.shape();

		if (shape.length == 4) {
			// Cnn输出结果
			long height = shape[2];
			long width = shape[3];
			long channel = shape[1];
			int imageType;
			if (channel == 1) {
				imageType = BufferedImage.TYPE_BYTE_GRAY;
			} else {
				imageType = BufferedImage.TYPE_INT_RGB;
			}
			image = new BufferedImage((int) width, (int) height, imageType);
			for (int x = 0; x < width; x++) {
				for (int y = 0; y < height; y++) {
					for (int band = 0; band < channel; band++) {
						double value = array.getDouble(0, band, y, x);
						value = unnorm(value);
						image.getRaster().setSample(x, y, band, value);
					}
				}
			}
		} else if (shape.length == 2) {
			int size = (int) shape[1];
			size = (int) Math.pow(size, 0.5);

			int width = size;
			int height = size;
			image = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
			for (int x = 0; x < width; x++) {
				for (int y = 0; y < height; y++) {
					double gray;
					if (shape[0] == width * height) {
						gray = array.getDouble(x + y * width, 0);
					} else if (shape[1] == width * height) {
						gray = array.getDouble(0, x + y * width);
					} else if (shape[0] == width && shape[1] == height) {
						gray = array.getDouble(y, x);
					} else {
						for (long s : shape) {
							System.out.println("s = " + s);
						}
						throw new IllegalArgumentException("shape.length = " + shape.length);
					}
					gray = unnorm(gray);
					image.getRaster().setSample(x, y, 0, (int) gray);
				}
			}
		}

		return image;
	}

	private static double unnorm(double value) {
		if (LowerBound == 0) {
			return value * 255;
		} else {
			return value * 127.5 + 127.5;
		}
	}

	private static double norm(double value) {
		if (LowerBound == 0) {
			return value / 255;
		} else {
			return (value - 127.5) / 127.5;
		}
	}

}
