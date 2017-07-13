package jp.co.example.dl.util;

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
		double[] y = new double[n];
		double max = 0;
		double sum = 0;

		for (int i = 0; i < n; i++) {
			if (max < x[i]) {
				max = x[i];
			}
		}

		for (int i = 0; i < n; i++) {
			y[i] = Math.exp(x[i] - max);
			sum += y[i];
		}

		for (int i = 0; i < n; i++) {
			y[i] /= sum;
		}

		return y;
	}

}
