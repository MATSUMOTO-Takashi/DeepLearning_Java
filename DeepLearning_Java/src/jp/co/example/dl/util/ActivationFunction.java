package jp.co.example.dl.util;

import java.util.stream.DoubleStream;

/**
 * 活性化関数を実装するクラス。
 *
 */
public class ActivationFunction {

	/**
	 * a=0のステップ関数。
	 *
	 * @param x 入力値
	 * @return 引数が0以上なら1を、そうでないなら-1を返す
	 */
	public static int step(double x) {
		return x >= 0 ? 1 : -1;
	}

	public static double[] softmax(double[] x, int n) {
		double max = DoubleStream.of(x).max().orElse(0);
		double[] y = DoubleStream.of(x).map(a -> Math.exp(a - max)).toArray();
		double sum = DoubleStream.of(y).sum();

		return DoubleStream.of(y).map(a -> a / sum).toArray();
	}

}
