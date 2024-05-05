use itertools::{iproduct, Itertools};
use proconio::input_interactive;
use rand::prelude::*;

/// エントリーポイント
fn main() {
    // input
    let input = Input::new();

    // solve
    let mut state = if input.m <= 2 {
        // n=20, m == 2 の時，n^(2m) でもok
        bayesian_solver::solve(&input)
    } else {
        naive_solver::solve(&input)
    };

    // check
    eprintln!(
        "# Score = {}, Cost = {}",
        state.eval_score(),
        state.eval_cost()
    );
}

/// 時計回り: Up, Right, Down, Left
const DYDX: [(usize, usize); 4] = [(!0, 0), (0, 1), (1, 0), (0, !0)];
/// マスの状況
#[derive(Clone, Debug, PartialEq)]
enum CellStatus {
    Unknown,
    Empty,
    Oil,
}

/// 入力によって一意に定まる情報
#[allow(dead_code)]
pub struct Input {
    /// 島の大きさ, 10 <= n <= 20
    n: usize,
    /// 油田の個数, 2 <= m <= 20
    m: usize,
    /// エラーパラメータ, 0.01 <= eps <= 0.2
    eps: f64,
    /// d(k) := k番目の油田の面積, 4 <= d <= n^2/m
    d: Vec<usize>,
    /// sum_d := d.sum()
    sum_d: u32,
    /// poly(k) := k番目の油田を左上基準になるように平行移動したときの座標集合
    poly: Vec<Vec<(usize, usize)>>,
    /// width(k) := k番目の油田の横幅
    width: Vec<usize>,
    /// height(k) := k番目の油田の縦幅
    height: Vec<usize>,
}
impl Input {
    fn new() -> Self {
        input_interactive! {
            n: usize,
            m: usize,
            eps: f64,
        }
        let mut d = Vec::with_capacity(m);
        let mut poly = Vec::with_capacity(m);
        let mut width = Vec::with_capacity(m);
        let mut height = Vec::with_capacity(m);
        for _ in 0..m {
            input_interactive! {
                d_k: usize,
                mut poly_k: [(usize, usize); d_k],
            }
            d.push(d_k);
            poly_k.sort();
            poly.push(poly_k.clone());
            let w = poly_k.iter().map(|&(_, x)| x).max().unwrap();
            let h = poly_k.iter().map(|&(y, _)| y).max().unwrap();
            width.push(w);
            height.push(h);
        }
        let sum_d = d.iter().sum::<usize>() as u32;
        Input {
            n,
            m,
            eps,
            d,
            sum_d,
            poly,
            width,
            height,
        }
    }
}

/// 解を表現する情報
#[derive(Clone, Debug)]
pub struct State {
    /// cell(y, x) := マス(y, x)の状態
    cell: Vec<Vec<CellStatus>>,
    /// v(y, x) := マス(y, x)の石油埋蔵量
    v: Vec<Vec<u32>>,
    sum_v: u32,
    interaction_count: usize,
    cost: f64,
    /// スコアは小さいほど良い
    score: i64,
}
impl State {
    fn new(input: &Input) -> Self {
        State {
            cell: vec![vec![CellStatus::Unknown; input.n]; input.n],
            v: vec![vec![0; input.n]; input.n],
            sum_v: 0,
            interaction_count: 0,
            cost: 0.0,
            score: 0,
        }
    }

    fn eval_cost(&self) -> f64 {
        self.cost
    }

    fn eval_score(&mut self) -> i64 {
        self.score = (self.cost * 1e6).round() as i64;
        self.score
    }

    /// op = 'q' or 'a'
    /// d = s.len()
    /// s = {(y, x), ...}
    fn interactive_io(&mut self, op: char, d: usize, s: &[(usize, usize)]) -> u32 {
        self.interaction_count += 1;
        let mut buf = Vec::with_capacity(d + 2);
        assert!(op == 'q' || op == 'a');
        buf.push(op.to_string());
        assert!(0 < d, "op={}, d={}, s={:?}", op, d, s);
        buf.push(format!("{}", d));
        for &(y, x) in s.iter() {
            buf.push(format!("{} {}", y, x));
        }
        let buf = buf.join(" ");
        println!("{}", buf);
        match op {
            'q' => {
                self.cost += 1.0 / f64::sqrt(d as f64);
                input_interactive! {
                    vs: u32,
                }
                vs
            }
            'a' => {
                input_interactive! {
                    status: u32,
                }
                if status == 0 {
                    self.cost += 1.0;
                }
                status
            }
            _ => unreachable!(),
        }
    }

    /// op = 'q' or 'a'
    fn interactive_io_wrapper(&mut self, op: char, y: usize, x: usize) -> u32 {
        let s = vec![(y, x)];
        self.interactive_io(op, 1, &s)
    }
}

/// コンテスト中に書いたやつ
/// グリッドサーチ+BFS
mod naive_solver {
    use super::*;

    /// (y, x) を含む油田をminingして確定する
    fn mining(state: &mut State, input: &Input, y: usize, x: usize) {
        // 幅優先だと，効率的に探索できそう
        // 深さ優先だと，形の特徴を掴みやすそうだけど，探索効率は悪そう
        let mut queue = std::collections::VecDeque::new();
        // let mut stack = Vec::new();
        for &(dy, dx) in DYDX.iter() {
            let ny = y.wrapping_add(dy);
            let nx = x.wrapping_add(dx);
            if ny >= input.n || nx >= input.n {
                continue;
            }
            if state.cell[ny][nx] == CellStatus::Unknown {
                queue.push_back((ny, nx));
                // stack.push((ny, nx));
            }
        }
        'mining: while let Some((y, x)) = queue.pop_front() {
            // 'mining: while let Some((y, x)) = stack.pop() {
            if state.cell[y][x] != CellStatus::Unknown {
                continue;
            }
            // (y, x) をmining
            let v = state.interactive_io_wrapper('q', y, x);
            println!("# queue={:?}", queue);
            // println!("# stack={:?}", stack);
            if v == 0 {
                state.cell[y][x] = CellStatus::Empty;
                continue;
            }
            state.cell[y][x] = CellStatus::Oil;
            println!("#c {} {} grey", y, x);
            state.v[y][x] = v;
            state.sum_v += v;
            if state.sum_v == input.sum_d {
                break 'mining;
            }
            for &(dy, dx) in DYDX.iter() {
                let ny = y.wrapping_add(dy);
                let nx = x.wrapping_add(dx);
                if ny >= input.n || nx >= input.n {
                    continue;
                }
                if state.cell[ny][nx] == CellStatus::Unknown {
                    queue.push_back((ny, nx));
                    // stack.push((ny, nx));
                }
            }
        }
    }

    /// 解説放送であったけど，真ん中の方から広げた方が良さそう
    /// ユーザー解説 (https://sankantsu.hatenablog.com/entry/2024/02/22/185034) の半々にしていくやつカッコ良い
    pub fn solve(input: &Input) -> State {
        let mut state = State::new(input);
        let grid_yx_4 = iproduct!((3..input.n).step_by(4), (3..input.n).step_by(4)).collect_vec();
        let grid_yx_3 = iproduct!((2..input.n).step_by(3), (2..input.n).step_by(3)).collect_vec();
        let grid_yx_2 = iproduct!((1..input.n).step_by(2), (1..input.n).step_by(2)).collect_vec();
        let all_yx = iproduct!(0..input.n, 0..input.n).collect_vec();
        let yx = [grid_yx_4, grid_yx_3, grid_yx_2, all_yx].concat();
        'mining: for &(y, x) in yx.iter() {
            if state.sum_v == input.sum_d {
                break 'mining;
            }
            if y == usize::MAX && x == usize::MAX {
                println!("#c 0 0 red");
                continue;
            }
            if state.cell[y][x] != CellStatus::Unknown {
                continue;
            }
            // (y, x) をmining
            let v = state.interactive_io_wrapper('q', y, x);
            if v == 0 {
                state.cell[y][x] = CellStatus::Empty;
                continue;
            }
            // (y, x) に油田を発見した場合
            state.cell[y][x] = CellStatus::Oil;
            println!("#c {} {} blue", y, x);
            state.v[y][x] = v;
            state.sum_v += v;
            if state.sum_v == input.sum_d {
                break 'mining;
            }
            self::mining(&mut state, input, y, x);
            if state.sum_v == input.sum_d {
                break 'mining;
            }
        }
        let s = iproduct!(0..input.n, 0..input.n)
            .filter(|&(y, x)| state.v[y][x] > 0)
            .collect_vec();
        let status = state.interactive_io('a', s.len(), &s);
        assert!(status == 1);

        state
    }
}

/// Mが小さい時限定のベイズ推定ソルバー
/// Mが大きい場合に対応したいなら，gen_all_state() せずに，焼きなましで状態を生成する必要がある．
/// 今回はベイズ推定を履修することが目的のため，そこまではやらない．
mod bayesian_solver {
    use super::*;

    /// 全ての盤面を生成
    /// v(i, y, x) := i番目の盤面における，(y, x)の油田埋蔵量
    /// ref: https://qiita.com/aplysia/items/c3f2111110ac5043710a
    fn gen_all_state(input: &Input) -> Vec<Vec<Vec<u32>>> {
        let mut v = Vec::new();
        let v0 = vec![vec![0; input.n]; input.n];
        v.push(v0);
        for poly_id in 0..input.m {
            let mut next_v = Vec::new();
            for state_i in v.iter() {
                for dy in 0..(input.n - input.height[poly_id]) {
                    for dx in 0..(input.n - input.width[poly_id]) {
                        let mut next_state_i = state_i.clone();
                        for &(y, x) in input.poly[poly_id].iter() {
                            next_state_i[y + dy][x + dx] += 1;
                        }
                        next_v.push(next_state_i);
                    }
                }
            }
            std::mem::swap(&mut next_v, &mut v);
        }
        v
    }

    /// 盤面b_iから大きさkのマス集合S_jを占い，
    /// 盤面b_iにおけるマス集合Sの埋蔵量がv_{b_i}(S_j) の時，
    /// マス集合S_jの占い結果がx_j，
    /// となる確率 P(k, v_{b_i}(S_j), x_j) を返す
    fn eval_prob(input: &Input, k: usize, vs: u32, x: u32) -> f64 {
        let (k, vs) = (k as f64, vs as f64);
        let mean = (k - vs) * input.eps + vs * (1.0 - input.eps);
        let std_dev = (k * input.eps * (1.0 - input.eps)).sqrt();
        let gauss = distributions::GaussianDistribution::new(mean, std_dev);
        match x {
            0 => {
                // (-inf, x+0.5)
                let x = x as f64;
                gauss.cdf(x + 0.5)
            }
            _ => {
                // [x-0.5, x+0.5)
                let x = x as f64;
                gauss.prob_in_range(x - 0.5, x + 0.5)
            }
        }
    }

    /// 確率を全部足して１になるように正規化
    fn normalize(p: &mut [f64]) {
        let sum = p.iter().sum::<f64>();
        assert!(sum > 0.0, "sum={}", sum);
        for pi in p.iter_mut() {
            *pi /= sum;
        }
    }

    /// ref: https://img.atcoder.jp/ahc030/ahc030.pdf
    ///
    /// 盤面全体の集合を B = {b_1, b_2, ...}，
    /// 占い対象のマス集合 S_1, S_2, ...に対して，得られた値が X = {x_1, x_2, ...}
    /// とする
    ///
    /// 盤面b_iから大きさkのマス集合S_jを占い，
    /// 盤面b_iにおけるマス集合Sの埋蔵量がv_{b_i}(S_j) の時，
    /// マス集合S_jの占い結果がx_j，
    /// となる確率を P(k, v_{b_i}(S_j), x_j) とする．
    /// 問題文より，この値は計算可能．
    ///
    /// 盤面b_iで，占い結果Xが得られる確率，つまり盤面b_iの尤度L_iは
    /// L_i = P(X|b_i) = \prod_{j} P(len(S_j), v_{b_i}(S_j), x_j)
    /// となる．
    ///
    /// 占い結果がXの時，盤面b_iが真の配置である確率，事後確率 P(b_i|X) はベイズの定理より
    /// P(b_i|X) = P(X|b_i) / \sum_j P(X|b_j)
    ///          = L_i / \sum_j L_j
    /// として計算可能．
    ///
    /// 今回は，尤度最大のiが分かればよいので，
    /// P(b_i|X) を求める必要は無く，L_i の最大を探せばよい．
    ///
    pub fn solve(input: &Input) -> State {
        let mut state = State::new(input);
        let mut rng = rand_pcg::Pcg64Mcg::new(42);
        let v = self::gen_all_state(input);
        let n = v.len();
        // p(b_i) := 盤面b_iの尤度
        let mut p = vec![1.0 / n as f64; n];
        // 全ての盤面のなかで最大の尤度が0.8なら，その盤面を答えてみる
        let p_threshold = 0.80;
        let ij = iproduct!(0..input.n, 0..input.n).collect_vec();
        let iter_limit = 2 * input.n * input.n;
        'main: while state.interaction_count < iter_limit {
            // TODO: kとsの選び方
            // 情報理論でやった，エントロピーとかを考えて良い感じにやりたい
            let k = input.sum_d as usize;
            let s = ij
                .choose_multiple(&mut rng, k)
                .map(|&(y, x)| (y, x))
                .collect_vec();
            println!("# io_cnt={}, k={}, s={:?}", state.interaction_count, k, s);
            let x = state.interactive_io('q', k, &s);
            for b_i in 0..n {
                let vs = s.iter().map(|&(y, x)| v[b_i][y][x]).sum();
                // 尤度
                p[b_i] *= self::eval_prob(input, k, vs, x);
            }
            self::normalize(&mut p);
            // f64::NAN のせいで，f64は普通にsortできない
            let max_p = *p.iter().max_by(|a, b| a.total_cmp(b)).unwrap();
            let max_p_i = p.iter().position(|x| *x == max_p).unwrap();
            println!(
                "# top 5 p={:?}",
                p.iter()
                    .sorted_by(|a, b| a.total_cmp(b))
                    .rev()
                    .take(5)
                    .collect_vec()
            );
            if max_p < p_threshold {
                continue 'main;
            }
            let s = iproduct!(0..input.n, 0..input.n)
                .filter(|&(y, x)| v[max_p_i][y][x] > 0)
                .collect_vec();
            let status = state.interactive_io('a', s.len(), &s);
            match status {
                0 => {
                    p[max_p_i] = 0.0;
                    continue 'main;
                }
                1 => {
                    break 'main;
                }
                _ => unreachable!(),
            }
        }
        state
    }
}

mod distributions {
    /// ガウス分布（正規分布）
    pub struct GaussianDistribution {
        mean: f64,
        std_dev: f64,
    }
    impl GaussianDistribution {
        /// mean: 平均
        /// std_dev: 標準偏差
        pub fn new(mean: f64, std_dev: f64) -> Self {
            Self { mean, std_dev }
        }

        /// Cumulative Distribution Function
        /// 累積分布関数
        /// x以下の値が出る確率を返す
        pub fn cdf(&self, x: f64) -> f64 {
            // libm::erf(x) := 2/sqrt(pi) * \int_{0}^{x} exp(-t^2) dt
            0.5 * (1.0 + libm::erf((x - self.mean) / (self.std_dev * std::f64::consts::SQRT_2)))
        }

        /// l以上r以下の値が出る確率を返す
        /// l, r は 閉区間でも開区間でもよい（連続確率分布において，ある一点の確率はゼロ）
        pub fn prob_in_range(&self, l: f64, r: f64) -> f64 {
            assert!(l <= r);
            if self.mean < l {
                return self.prob_in_range(2.0 * self.mean - r, 2.0 * self.mean - l);
            }
            self.cdf(r) - self.cdf(l)
        }
    }

    #[cfg(test)]
    mod test {
        #[test]
        fn test_gauss_cdf() {
            // 標準正規分布 N(0, 1)
            let (mean, std_dev) = (0.0, 1.0);
            let gauss = super::GaussianDistribution::new(mean, std_dev);
            let p = gauss.cdf(0.0);
            let correct_p = 0.5;
            // 正解との誤差がeps以下ならok
            let eps = 1e-5;
            assert!((p - correct_p).abs() < eps);
        }

        #[test]
        fn test_gauss_prob_in_range() {
            // 標準正規分布 N(0, 1)
            let (mean, std_dev) = (0.0, 1.0);
            let gauss = super::GaussianDistribution::new(mean, std_dev);
            let p = gauss.prob_in_range(-1.0, 1.0);
            let correct_p = 0.6826894921370859;
            // 正解との誤差がeps以下ならok
            let eps = 1e-5;
            assert!((p - correct_p).abs() < eps);
        }
    }
}
