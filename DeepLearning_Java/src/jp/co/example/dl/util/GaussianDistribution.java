package jp.co.example.dl.util;

import java.util.Random;

/**
 * 正規分布（ガウス分布）に従った値を返す、サンプルデータ生成用のクラス。
 *
 */
public class GaussianDistribution {
	private final double mean;
	private final double var;
	private final Random rng; // random number generator

	public GaussianDistribution(double mean, double var) {
		this(mean, var, null);
	}

	public GaussianDistribution(double mean, double var, Random rng) {
		if (var < 0) {
			throw new IllegalArgumentException("Variance must be non-negative value.");
		}

		this.mean = mean;
		this.var = var;

		if (rng == null) {
			rng = new Random();
		}
		this.rng = rng;
	}

	public double random() {
		double r = 0;
		while (r == 0) {
			r = rng.nextDouble();
		}

		double c = Math.sqrt(-2 * Math.log(r));

		if (rng.nextDouble() < 0.5) {
			return c * Math.sin(2 * Math.PI * rng.nextDouble()) * var + mean;
		} else {
			return c * Math.cos(2 * Math.PI * rng.nextDouble()) * var + mean;
		}
	}
}
