package jp.co.example.dl.SingleLayerNeuralNetworks;

import java.util.Random;

import jp.co.example.dl.util.ActivationFunction;
import jp.co.example.dl.util.GaussianDistribution;

/**
 * 単純パーセプトロン（単層ニューラルネットワーク）を実装するクラス。
 *
 */
public class Perceptrons {
	private int nIn; // 入力データの次元数
	private double[] w; // パーセプトロンの重み

	public Perceptrons(int nIn) {
		this.nIn = nIn;
		w = new double[nIn];
	}

	/**
	 * 学習を行います。
	 * パーセプトロンの学習とはすなわち、正しい重みを探し出すことです。
	 *
	 * @param x トレーニングデータ
	 * @param t ラベル値（-1 または 1）
	 * @param learningRate 学習率
	 * @return 正しく分類できていれば1、そうでなければ0を返します。
	 */
	public int train(double[] x, int t, double learningRate) {
		int classified = 0;
		double c = 0;

		// データが正しく分類されているかのチェック
		for (int i = 0; i < nIn; i++) {
			c += w[i] * x[i] * t;
		}

		if (c > 0) {
			// 正しく分類できていれば1を返す
			classified = 1;
		} else {
			// 正しく分類できていなければ勾配降下法を適用
			for (int i = 0; i < nIn; i++) {
				w[i] += learningRate * x[i] * t;
			}
		}

		return classified;
	}

	/**
	 * ネットワークを通じて予測を返す。
	 *
	 * @param x データ
	 * @return パーセプトロンによる予測結果
	 */
	public int predict(double[] x) {
		double preActivation = 0;
		for (int i = 0; i < nIn; i++) {
			preActivation += w[i] * x[i];
		}

		return ActivationFunction.step(preActivation);
	}

	/**
	 * エントリーポイント。
	 *
	 * @param args
	 */
	public static void main(String[] args) {
		final int train_N = 1000;	// トレーニングデータ数
		final int test_N = 200;	// テストデータ数
		final int nIn = 2;			// 入力層のユニット数（ニューロン数）

		double[][] train_X = new double[train_N][nIn];	// トレーニングデータ
		int[] train_T = new int[train_N];					// トレーニングデータのラベル

		double[][] test_X = new double[test_N][nIn];	// テストデータ
		int[] test_T = new int[test_N];				// テストデータのラベル
		int[] predicated_T = new int[test_N];		// モデルの予測結果

		final int epochs = 2000;			// トレーニング数の上限（トレーニングを繰り返す回数のことをエポックと呼ぶ）
		final double learningRate = 1;	// 学習率はパーセプトロンでは1でよい

		// サンプルデータ用のジェネレーター
		final Random rng = new Random(1234);
		GaussianDistribution g1 = new GaussianDistribution(-2, 1, rng);
		GaussianDistribution g2 = new GaussianDistribution(+2, 1, rng);

		// データセットの生成
		// 最初の500件はクラス1に属する（[-2.0, 2.0]の近辺に分布する）
		// クラス1のラベル値は1となる
		for (int i = 0; i < train_N / 2 - 1; i++) {
			train_X[i][0] = g1.random();
			train_X[i][1] = g2.random();
			train_T[i] = 1;
		}

		// 次の500件はクラス2に属する（[2.0, -2.0]の近辺に分布する）
		// クラス2のラベル値は-1となる
		for (int i = train_N / 2; i < train_N; i++) {
			train_X[i][0] = g2.random();
			train_X[i][1] = g1.random();
			train_T[i] = -1;
		}

		// テストデータも同様に生成する
		for (int i = 0; i < test_N / 2 - 1; i++) {
			test_X[i][0] = g1.random();
			test_X[i][1] = g2.random();
			test_T[i] = 1;
		}

		for (int i = test_N / 2; i < test_N; i++) {
			test_X[i][0] = g2.random();
			test_X[i][1] = g1.random();
			test_T[i] = -1;
		}

		// パーセプトロンモデルの構築
		int epoch = 0;
		Perceptrons classifier = new Perceptrons(nIn);

		// トレーニング
		while (epoch < epochs) {
			int classified = 0;
			for (int i = 0; i < train_N; i++) {
				classified += classifier.train(train_X[i], train_T[i], learningRate);
			}

			if (classified == train_N) break; // 全データが正しく分類されていたら学習をストップ
			epoch++;
		}

		// 構築したパーセプトロンを使ってのテスト
		for (int i = 0; i < test_N; i++) {
			// テストデータの分類結果は配列に格納しておく
			predicated_T[i] = classifier.predict(test_X[i]);
		}

		// モデルの評価
		int[][] confusionMatrix = new int[2][2];	// 混合行列
		double accuracy = 0;							// 正解率
		double precision = 0;							// 精度
		double recall = 0;							// 再現率

		for (int i = 0; i < test_N; i++) {
			if (predicated_T[i] > 0) {
				if (test_T[i] > 0) {
					accuracy++;
					precision++;
					recall++;
					confusionMatrix[0][0]++;
				} else {
					confusionMatrix[1][0]++;
				}
			} else {
				if (test_T[i] > 0) {
					confusionMatrix[0][1]++;
				} else {
					accuracy++;
					confusionMatrix[1][1]++;
				}
			}
		}

		accuracy /= test_N;
		precision /= confusionMatrix[0][0] + confusionMatrix[1][0];
		recall /= confusionMatrix[0][0] + confusionMatrix[0][1];

		System.out.println("----------------------------");
		System.out.println("Perceptrons model evaluation");
		System.out.println("----------------------------");
		System.out.printf("Accuracy:  %.1f %%\n", accuracy * 100);
		System.out.printf("Precision: %.1f %%\n", precision * 100);
		System.out.printf("Recall:    %.1f %%\n", recall * 100);
	}

}
