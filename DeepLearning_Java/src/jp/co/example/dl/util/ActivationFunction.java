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

}
