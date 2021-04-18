package com.pan.test;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.List;

import javax.imageio.ImageIO;

import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import lombok.Setter;

public class GanMultiDataSetIterator implements MultiDataSetIterator {

	public static double LowerBound = 0; // -1

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	@Setter
	private boolean flag = false;

	private int cursor = 0;
	private final int batchSize;
	private final int height;
	private final int width;
	private final int channels;
	private final int vectorSize;

	private DataType dataType = DataType.FLOAT;

	protected BaseImageLoader imageLoader;

	private MultiDataSetPreProcessor preProcessor;

	private final List<File> dataList;

	public GanMultiDataSetIterator(int batchSize, int height, int width, int channels, int vectorSize,
			List<File> dataList) {
		this.batchSize = batchSize;
		this.height = height;
		this.width = width;
		this.channels = channels;
		this.vectorSize = vectorSize;
		this.dataList = dataList;

		imageLoader = new NativeImageLoader(this.height, this.width, this.channels);
	}

	public GanMultiDataSetIterator(int batchSize, int height, int width, int channels, int vectorSize,
			List<File> dataList, DataType dataType) {
		this.batchSize = batchSize;
		this.height = height;
		this.width = width;
		this.channels = channels;
		this.vectorSize = vectorSize;
		this.dataList = dataList;
		this.dataType = dataType;

		imageLoader = new NativeImageLoader(this.height, this.width, this.channels);
	}

	@Override
	public boolean hasNext() {
		return cursor < dataList.size();
	}

	@Override
	public MultiDataSet next() {
		return next(this.batchSize);
	}

	@Override
	public MultiDataSet next(int num) {
		int end = cursor + num;

		if (end >= dataList.size()) {
			num = dataList.size() - cursor;
			end = dataList.size();
		}

		INDArray z = Nd4j.randn(0, 1, new long[] { num, vectorSize }, Nd4j.getRandom());
		INDArray x = Nd4j.create(dataType, new long[] { num, this.channels, this.height, this.width });

		INDArray output_Dx = Nd4j.zeros(dataType, new long[] { num, 2 });
		INDArray output_Dgz = Nd4j.zeros(dataType, new long[] { num, 2 });

		for (int i = 0; i < num; i++) {
			File image = dataList.get(i);
			try {
				x.put(new INDArrayIndex[] { NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.all(),
						NDArrayIndex.all() }, imageLoader.asMatrix(image));
				Transforms.sigmoid(x, false);
			} catch (Exception | Error e) {
				e.printStackTrace();
			}

			output_Dx.putScalar(new int[] { i, 1 }, 1);

			if (flag) {
				output_Dgz.putScalar(new int[] { i, 1 }, 1);
			} else {
				output_Dgz.putScalar(new int[] { i, 0 }, 1);
			}

			cursor++;
		}

		MultiDataSet ds = new MultiDataSet(new INDArray[] { x, z }, new INDArray[] { output_Dx, output_Dgz });

		if (preProcessor != null) {
			preProcessor.preProcess(ds);
		}

		return ds;
	}

	@Override
	public void setPreProcessor(MultiDataSetPreProcessor preProcessor) {
		this.preProcessor = preProcessor;
	}

	@Override
	public MultiDataSetPreProcessor getPreProcessor() {
		return this.preProcessor;
	}

	@Override
	public boolean resetSupported() {
		return true;
	}

	@Override
	public boolean asyncSupported() {
		return false;
	}

	@Override
	public void reset() {
		cursor = 0;
	}

	protected INDArray readImage(File image, int width, int height, int channel, int demension) {
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
						// 错误：
						// values[(i * height + j) * channel] = rValue;
						// 正确：
						values[(i * height + j)] = rValue;

						if (channel == 3) {
							// 如果是彩色图，还需要设置g值和b值
							double gValue = norm(g);
							double bValue = norm(b);
							// 错误：
							// values[(i * height + j) * channel + 1] = gValue;
							// values[(i * height + j) * channel + 2] = bValue;
							// 正确：
							values[(i * height + j) + 1 * width * height] = gValue;
							values[(i * height + j) + 2 * width * height] = bValue;
						}
					}
				}

				if (demension == 4) {
					indArray = Nd4j.create(values, new int[] { 1, channel, width, height });
				} else if (demension == 2) {
					indArray = Nd4j.create(values, new int[] { 1, channel * width * height });
				}
			} catch (IOException e) {

			}
		}

		return indArray;
	}

	private double norm(double value) {
		if (LowerBound == 0) {
			return value / 255;
		} else {
			return (value - 127.5) / 127.5;
		}
	}

}
