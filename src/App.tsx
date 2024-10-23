import { useState, useEffect } from "react";
import init, { run } from "../wasm/pkg/wasm";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { coldarkDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableContainer from "@mui/material/TableContainer";
import TableHead from "@mui/material/TableHead";
import TableRow from "@mui/material/TableRow";
import Paper from "@mui/material/Paper";
import { ThemeProvider, createTheme } from "@mui/material/styles";

function App() {
  const [wasmLoaded, setWasmLoaded] = useState(false);
  const [csvData, setCsvData] = useState<string | null>(null);
  const [executionTime, setExecutionTime] = useState<number | null>(null);
  const [result, setResult] = useState<string | null>(null);

  useEffect(() => {
    init().then(() => setWasmLoaded(true));
    fetch("/Iris.csv")
      .then((response) => response.text())
      .then((data) => {
        // 取得したCSVデータをstringにしてからsetする
        setCsvData(data);
        console.log(data);
      })
      .catch((error) => {
        console.error("CSVの取得中にエラーが発生しました:", error);
      });
  }, []);

  const handleRunAnalysis = () => {
    if (wasmLoaded && csvData) {
      const startTime = performance.now();
      const result = run(csvData);
      const endTime = performance.now();
      const executionTime = endTime - startTime;
      setExecutionTime(executionTime);
      setResult(result);
    }
  };

  const parseCsvData = (data: string) => {
    const rows = data.split("\n");
    const headers = rows[0].split(",");
    const parsedData = rows.slice(1).map((row) => {
      const values = row.split(",");
      return headers.reduce((obj, header, index) => {
        obj[header] = values[index];
        return obj;
      }, {} as Record<string, string>);
    });
    return { headers, parsedData };
  };

  const { headers, parsedData } = csvData
    ? parseCsvData(csvData)
    : { headers: [], parsedData: [] };

  const darkTheme = createTheme({
    palette: {
      mode: "dark",
    },
  });

  return (
    <ThemeProvider theme={darkTheme}>
      <div style={{ margin: "20px" }}>
        <h1>爆速WebAssembly機械学習</h1>
        <section>
          <h2>アプリの概要</h2>
          <p>
            このアプリでは、フロント技術だけでサーバーレスにアイリスデータセットの分類を行うことができます。もちろんスマートフォンでも動作可能です。
          </p>
          <p>
            実際にサーバーサイドは存在しないため、メンテナンスの必要がありません。
          </p>
        </section>
        <section>
          <h2>技術の魅力</h2>
          <p>
            WebAssemblyとRustを使って、アルゴリズムをスクラッチで実装しました。
            WebAssemblyははるかに機械コードに近い形で実行可能なため、非常に高速です。例えば、WebAssemblyで直接記述する場合、対応する型は整数と浮動小数点数のみです(今回はRustを用いて記述したため、気にする必要はありませんが)。
          </p>
          <p>
            主に画像処理などの計算量が多い処理に適しており、Figmaなどのデザインツールや、Google Meetのモザイク処理などに利用されています。
          </p>
        </section>
        <section>
          <h2>ロジスティック回帰の実装</h2>
          <p>
            今回はRustを用いてロジスティック回帰を実装しました。
          </p>
          <p>
            今回はコンパイルしたため、実際に動作するわけではありませんが、Rustではでは「所有権」という概念を用いてメモリ管理を行なっており、厳密な型システムによって非常に高速に安全に分かりやすくコードを書くことができます。
          </p>
          <p>↓下記は実際にRustで実装したロジスティック回帰です。</p>
          <SyntaxHighlighter
            language="rust"
            style={coldarkDark}
            className="code-block"
          >
            {`fn logistic_regression(
    features: &Vec<Vec<f64>>,
    labels: &Vec<f64>,
    learning_rate: f64,
    epochs: usize,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n_samples = features.len();
    let n_features = features[0].len();
    let n_classes = 3;

    let mut weights = vec![vec![0.0; n_features]; n_classes];
    let mut biases = vec![0.0; n_classes];

    for _ in 0..epochs {
        let mut dw = vec![vec![0.0; n_features]; n_classes];
        let mut db = vec![0.0; n_classes];

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

        for k in 0..n_classes {
            for j in 0..n_features {
                weights[k][j] -= learning_rate * dw[k][j] / n_samples as f64;
            }
            biases[k] -= learning_rate * db[k] / n_samples as f64;
        }
    }

    (weights, biases)
}`}
          </SyntaxHighlighter>
        </section>
        <section>
          <h2>アイリスデータセット</h2>
          <p>お馴染みのアイリスデータセットです。</p>
          <TableContainer component={Paper} style={{ maxHeight: 300 }}>
            <Table stickyHeader>
              <TableHead>
                <TableRow>
                  {headers.map((header) => (
                    <TableCell key={header}>{header}</TableCell>
                  ))}
                </TableRow>
              </TableHead>
              <TableBody>
                {parsedData.map((row, index) => (
                  <TableRow key={index}>
                    {headers.map((header) => (
                      <TableCell key={header}>{row[header]}</TableCell>
                    ))}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </section>
        <section>
          <h2>実際に実行する</h2>
          <p>
            分かりやすいよう、わざと計算に時間をかけるため100,000エポックにしています。
          </p>
        </section>
        <button onClick={handleRunAnalysis} disabled={!wasmLoaded || !csvData}>
          学習と予測を実行
        </button>
        {executionTime !== null && (
          <p>実行時間: {(executionTime / 1000).toFixed(4)} 秒</p>
        )}
        {result !== null && <p>{result}</p>}
        <p>
          是非お手持ちのPython環境でも同じようにお試しください。私の環境では100,000エポックで6.5352秒ほどでした。
        </p>
        <p>
          Pythonの内部はC言語で実装されているため、かなり高速のはずですが、Web技術のみを用いてるのにも関わらず、それよりも高速に動作します。
        </p>
        <h2>使用した技術</h2>
        <ul>
          <li>Vite</li>
          <li>React</li>
          <li>WebAssembly</li>
          <li>Rust</li>
          <li>TypeScript</li>
        </ul>
      </div>
    </ThemeProvider>
  );
}

export default App;
