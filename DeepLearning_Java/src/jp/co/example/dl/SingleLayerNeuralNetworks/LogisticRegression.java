package jp.co.example.dl.SingleLayerNeuralNetworks;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import jp.co.example.dl.util.ActivationFunction;
import jp.co.example.dl.util.GaussianDistribution;

/**
 * 多クラスロジスティック回帰を実装するクラス。
 *
 */
public class LogisticRegression {
	private int nIn;
	private int nOut;
	private double[][] W;	// ネットワークの重み
	private double[] b;	// バイアス

	public LogisticRegression(int nIn, int nOut) {
		this.nIn = nIn;
		this.nOut = nOut;

		W = new double[nOut][nIn];
		b = new double[nOut];
	}

	public double[][] train(double[][] X, int[][] T, int minibatchSize, double learningRate) {
		double[][] grad_W = new double[nOut][nIn];
		double[] grad_b = new double[nOut];

		double[][] dY = new double[minibatchSize][nOut]; // 予測データと正しいデータとの誤差

		// SGDによるトレーニング
		// 1. 重みWとバイアスbの勾配を計算
		for (int n = 0; n < minibatchSize; n++) {
			double [] predicated_Y = output(X[n]);

			for (int j = 0; j < nOut; j++) {
				dY[n][j] = predicated_Y[j] - T[n][j];

				for (int i = 0; i < nIn; i++) {
					grad_W[j][i] += dY[n][j] * X[n][i];
				}

				grad_b[j] += dY[n][j];
			}
		}

		// 2. パラメーターを更新
		for (int j = 0; j < nOut; j++) {
			for (int i = 0; i < nIn; i++) {
				W[j][i] -= learningRate * grad_W[j][i] / minibatchSize;
			}
			b[j] -= learningRate * grad_b[j] / minibatchSize;
		}

		return dY;
	}

	public Integer[] predict(double[] x) {
		// 入力データを学習済みモデルによって活性化し、出力層へ
		double[] y = output(x);

		int argmax = -1;
		double max = 0;

		for (int i = 0; i < nOut; i++) {
			if (max < y[i]) {
				max = y[i];
				argmax = i;
			}
		}

		// 出力は確率で表されているため、ラベル（分類されたクラス）に変換
		// 最大値のインデックスには1を、それ以外は0をセット
		Integer[] t = new Integer[nOut];
		Arrays.fill(t, 0);
		t[argmax] = 1;

		return t;
	}

	private double[] output(double[] x) {
		double[] preActivation = new double[nOut];

		for (int j = 0; j < nOut; j++) {
			for (int i = 0; i < nIn; i++) {
				preActivation[j] += W[j][i] * x[i];
			}

			preActivation[j] += b[j]; // 線形活性にも対応させている
		}

		return ActivationFunction.softmax(preActivation);
	}

	public static void main(String[] args) {
		final Random rng = new Random(1234); // seed

		final int patterns = 3;				// クラス数
		final int train_N = 400 * patterns;	// トレーニングデータ数
		final int test_N = 60 * patterns;		// テストデータ数
		final int nIn = 2;						// 入力層のユニット数
		final int nOut = patterns;

		// トレーニングデータ
		double[][] train_X = new double[train_N][nIn];
		int[][] train_T = new int[train_N][nOut];

		// テストデータ
		double[][] test_X = new double[test_N][nIn];
		Integer[][] test_T = new Integer[test_N][nOut];
		Integer[][] predicated_T = new Integer[test_N][nOut];

		final int epochs = 2000;	// トレーニング回数の上限
		final double learningRate = 0.2;

		final int minibatchSize = 50;					// 各ミニバッチに含まれるデータ数
		final int minibatch_N = train_N / minibatchSize;	// ミニバッチ数

		// 入力データであるトレーニングデータのミニバッチ
		double[][][] train_X_minibatch = new double[minibatch_N][minibatchSize][nIn];
		// トレーニングデータの出力
		int[][][] train_T_minibatch = new int[minibatch_N][minibatchSize][nOut];
		// SGD（Stochastic Gradient Descent：確率的勾配降下法）が適用されるミニバッチのインデックス（順番）
		List<Integer> minibatchIndex = IntStream.range(0, train_N).mapToObj(Integer::valueOf).collect(Collectors.toList());
		Collections.shuffle(minibatchIndex);

		// サンプルデータの生成
		GaussianDistribution g1 = new GaussianDistribution(-2, 1, rng);
		GaussianDistribution g2 = new GaussianDistribution(+2, 1, rng);
		GaussianDistribution g3 = new GaussianDistribution( 0, 1, rng);

		// クラス1のデータセット
		for (int i = 0; i < train_N / patterns - 1; i++) {
			train_X[i][0] = g1.random();
			train_X[i][1] = g2.random();
			train_T[i] = new int[] { 1, 0, 0 };
		}

		// クラス2のデータセット
		for (int i = train_N / patterns - 1; i < train_N / patterns * 2 - 1; i++) {
			train_X[i][0] = g2.random();
			train_X[i][1] = g1.random();
			train_T[i] = new int[] { 0, 1, 0 };
		}

		// クラス3のデータセット
		for (int i = train_N / patterns * 2 - 1; i < train_N; i++) {
			train_X[i][0] = g3.random();
			train_X[i][1] = g3.random();
			train_T[i] = new int[] { 0, 0, 1 };
		}

		// テストデータも同様に
		for (int i = 0; i < test_N / patterns - 1; i++) {
			test_X[i][0] = g1.random();
			test_X[i][1] = g2.random();
			test_T[i] = new Integer[] { 1, 0, 0 };
		}

		for (int i = test_N / patterns - 1; i < test_N / patterns * 2 - 1; i++) {
			test_X[i][0] = g2.random();
			test_X[i][1] = g1.random();
			test_T[i] = new Integer[] { 0, 1, 0 };
		}

		// クラス3のデータセット
		for (int i = test_N / patterns * 2 - 1; i < test_N; i++) {
			test_X[i][0] = g3.random();
			test_X[i][1] = g3.random();
			test_T[i] = new Integer[] { 0, 0, 1 };
		}

		// ミニバッチ用のトレーニングデータ生成
		for (int i = 0; i < minibatch_N; i++) {
			for (int j = 0; j < minibatchSize; j++) {
				train_X_minibatch[i][j] = train_X[minibatchIndex.get(i * minibatchSize + j)];
				train_T_minibatch[i][j] = train_T[minibatchIndex.get(i * minibatchSize + j)];
			}
		}

		// ロジスティック回帰モデルの構築
		LogisticRegression classifier = new LogisticRegression(nIn, nOut);

		// トレーニング
		double _learningRate = learningRate;
		for (int epoch = 0; epoch < epochs; epoch++) {
			for (int batch = 0; batch < minibatch_N; batch++) {
				classifier.train(train_X_minibatch[batch], train_T_minibatch[batch], minibatchSize, _learningRate);
			}
			_learningRate *= 0.95;
		}

		// 構築したロジスティック回帰モデルを使ってのテスト
		for (int i = 0; i < test_N; i++) {
			predicated_T[i] = classifier.predict(test_X[i]);
		}

		// モデルの評価
		int[][] confusionMatrix = new int[patterns][patterns];
		double accuracy = 0;
		double[] precision = new double[patterns];
		double[] recall = new double[patterns];

		for (int i = 0; i < test_N; i++) {
			int predicated = Arrays.asList(predicated_T[i]).indexOf(1);
			int actual = Arrays.asList(test_T[i]).indexOf(1);

			confusionMatrix[actual][predicated]++;
		}

		for (int i = 0; i < patterns; i++) {
			double col = 0;
			double row = 0;

			for (int j = 0; j < patterns; j++) {
				if (i == j) {
					accuracy += confusionMatrix[i][j];
					precision[i] += confusionMatrix[j][i];
					recall[i] += confusionMatrix[i][j];
				}

				col += confusionMatrix[j][i];
				row += confusionMatrix[i][j];
			}

			precision[i] /= col;
			recall[i] /= row;
		}

		accuracy /= test_N;

		System.out.println("------------------------------------");
		System.out.println("Logistic Regression model evaluation");
		System.out.println("------------------------------------");
		System.out.printf("Accuracy: %.1f %%\n", accuracy * 100);
		System.out.println("Precision:");
		for (int i = 0; i < patterns; i++) {
			System.out.printf(" class %d: %.1f %%\n", i + 1, precision[i] * 100);
		}
		System.out.println("Recall:");
		for (int i = 0; i < patterns; i++) {
			System.out.printf(" class %d: %.1f %%\n", i + 1, recall[i] * 100);
		}
	}

}
