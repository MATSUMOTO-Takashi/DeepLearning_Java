package jp.co.example.dl.util;

/**
 * 活性化関数を実装するクラス。
 *
 */
public class ActivationFunction {

	/**
	 * パーセプトロンのステップ関数。
	 *
	 * @param x 素性ベクトルのそれぞれの要素と重さを掛け合わせたものの総和
	 * @return 活性化した値
	 */
	public static int step(double x) {
		return x >= 0 ? 1 : -1;
	}

}
