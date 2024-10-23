use wasm_bindgen::prelude::*;
use rand::seq::SliceRandom;

// エントリーポイント
#[wasm_bindgen]
pub fn run(csv_data: &str) -> String {
    // データを読み込む
    let (features, labels) = load_data(csv_data).unwrap();

    // データを訓練セットとテストセットに分割する
    let (train_features, train_labels, test_features, test_labels) = split_data(&features, &labels, 0.8);

    // ロジスティック回帰モデルを訓練する
    let (weights, biases) = logistic_regression(&train_features, &train_labels, 0.1, 100000);

    // テストデータで予測する
    let predictions = predict(&test_features, &weights, &biases);

    // テストデータでの正解率を計算して返り値として返す
    let accuracy = compute_accuracy(&predictions, &test_labels);
    format!("Test Accuracy: {}%", accuracy * 100.0).into()
}

// データを読み込む関数
fn load_data(data: &str) -> Result<(Vec<Vec<f64>>, Vec<f64>), Box<dyn std::error::Error>> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(data.as_bytes());
    
    let mut features: Vec<Vec<f64>> = Vec::new();
    let mut labels: Vec<f64> = Vec::new();

    for result in rdr.records() {
        let record = result?;
        if record.len() < 6 {
            return Err("各行には少なくとも6つのフィールドが必要です".into());
        }

        // IDをスキップし、特徴量を取得
        let feature: Vec<f64> = record.iter().skip(1).take(4)
            .map(|v| v.parse::<f64>())
            .collect::<Result<Vec<f64>, _>>()?;
        
        features.push(feature);

        // ラベルを文字列から数値に変換
        let label = match record.get(5).ok_or("ラベルが見つかりません")?.as_ref() {
            "Iris-setosa" => 0.0,
            "Iris-versicolor" => 1.0,
            "Iris-virginica" => 2.0,
            unknown => return Err(format!("不明なラベル: {}", unknown).into()),
        };
        labels.push(label);
    }

    Ok((features, labels))
}

// データを訓練セットとテストセットに分割する関数
fn split_data(features: &Vec<Vec<f64>>, labels: &Vec<f64>, train_ratio: f64) -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>, Vec<f64>) {
    let train_size = (features.len() as f64 * train_ratio) as usize;
    let mut rng = rand::thread_rng();
    let mut indices: Vec<usize> = (0..features.len()).collect();
    indices.shuffle(&mut rng);

    let train_indices = &indices[..train_size];
    let test_indices = &indices[train_size..];

    let train_features: Vec<Vec<f64>> = train_indices.iter().map(|&i| features[i].clone()).collect();
    let train_labels: Vec<f64> = train_indices.iter().map(|&i| labels[i]).collect();
    let test_features: Vec<Vec<f64>> = test_indices.iter().map(|&i| features[i].clone()).collect();
    let test_labels: Vec<f64> = test_indices.iter().map(|&i| labels[i]).collect();

    (train_features, train_labels, test_features, test_labels)
}

// ロジスティック回帰の訓練関数（多クラス分類用に修正）
fn logistic_regression(
    features: &Vec<Vec<f64>>,
    labels: &Vec<f64>,
    learning_rate: f64,
    epochs: usize,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n_samples = features.len();
    let n_features = features[0].len();
    let n_classes = 3; // Irisデータセットは3クラス

    // 重みとバイアスを初期化
    let mut weights = vec![vec![0.0; n_features]; n_classes];
    let mut biases = vec![0.0; n_classes];

    // 繰り返し学習
    for _ in 0..epochs {
        let mut dw = vec![vec![0.0; n_features]; n_classes];
        let mut db = vec![0.0; n_classes];

        // 勾配を計算
        for i in 0..n_samples {
            let mut scores = vec![0.0; n_classes];
            for k in 0..n_classes {
                scores[k] = dot_product(&features[i], &weights[k]) + biases[k];
            }
            let probs = softmax(&scores);

            for k in 0..n_classes {
                let error = if labels[i] == k as f64 { probs[k] - 1.0 } else { probs[k] };
                for j in 0..n_features {
                    dw[k][j] += error * features[i][j];
                }
                db[k] += error;
            }
        }

        // パラメータを更新
        for k in 0..n_classes {
            for j in 0..n_features {
                weights[k][j] -= learning_rate * dw[k][j] / n_samples as f64;
            }
            biases[k] -= learning_rate * db[k] / n_samples as f64;
        }
    }

    (weights, biases)
}

// 予測関数（多クラス分類用に修正）
fn predict(features: &Vec<Vec<f64>>, weights: &Vec<Vec<f64>>, biases: &Vec<f64>) -> Vec<f64> {
    let mut predictions = Vec::new();

    for x in features {
        let mut scores = vec![0.0; weights.len()];
        for k in 0..weights.len() {
            scores[k] = dot_product(x, &weights[k]) + biases[k];
        }
        let probs = softmax(&scores);
        let class = probs.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as f64;
        predictions.push(class);
    }

    predictions
}

// ソフトマックス関数（新規追加）
fn softmax(scores: &Vec<f64>) -> Vec<f64> {
    let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_scores: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp_scores: f64 = exp_scores.iter().sum();
    exp_scores.iter().map(|&s| s / sum_exp_scores).collect()
}

// 正解率を計算する関数
fn compute_accuracy(predictions: &Vec<f64>, labels: &Vec<f64>) -> f64 {
    let mut correct = 0;
    for (pred, label) in predictions.iter().zip(labels.iter()) {
        if pred == label {
            correct += 1;
        }
    }
    correct as f64 / labels.len() as f64
}

// 内積を計算する関数
fn dot_product(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}
