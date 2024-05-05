use grid::Grid;
use itertools::Itertools;
use proconio::input;
use rand::prelude::*;
use std::time::Instant;

fn main() {
    let input = Input::new();
    let hparam = HyperParametor::new();
    let output = solver::solve(&input, &hparam);
    output.write(&input);
}

/// 入力によって一意に定まる情報
#[allow(dead_code)]
pub struct Input {
    /// start_time
    start_time: Instant,
    /// n: グリッドの大きさ, n = 50
    n: usize,
    /// n2: パディングを含めたグリッドの大きさ, N = 52
    n2: usize,
    /// m: 区の数, m = 100
    m: usize,
    /// m1: 外周を含めた区の数, M = 101
    m1: usize,
    /// c(y, x): マス(y, x)の色, 0 <= c(y, x) <= m, 外周はパディング
    c: Grid<usize>,
    /// adj(c1, c2): 区c1とc2が隣接しているかどうか
    adj: Grid<bool>,
    /// around(c1): 区c1の周囲の区のリスト
    around: Vec<Vec<usize>>,
}
impl Input {
    fn new() -> Self {
        let start_time = Instant::now();
        input! {
            n: usize, m: usize,
            c: [[usize; n]; n],
        }
        let mut cc = Grid::with_default(n + 2, n + 2);
        for y in 0..n {
            for x in 0..n {
                cc[(y + 1, x + 1)] = c[y][x];
            }
        }
        let mut adj = Grid::with_default(m + 1, m + 1);
        let mut around = vec![vec![]; m + 1];
        for y in 0..=n {
            for x in 0..=n {
                let c1 = c[y][x];
                let nynx = adj.nynx4(y, x);
                for (ny, nx) in nynx {
                    let c2 = cc[(ny, nx)];
                    adj[(c1, c2)] = true;
                    adj[(c2, c1)] = true;
                    around[c1].push(c2);
                    around[c2].push(c1);
                }
            }
        }
        let around = around
            .iter_mut()
            .map(|v| v.clone().into_iter().sorted().dedup().collect_vec())
            .collect_vec();
        Input {
            start_time,
            n,
            n2: n + 2,
            m,
            m1: m + 1,
            c: cc,
            adj,
            around,
        }
    }
}

pub struct HyperParametor {
    /// 実行時間制限 (msec)
    time_limit: f64,
    /// 焼きなましの開始温度
    start_temp: f64,
    /// 焼きなましの終了温度
    end_temp: f64,
}
impl HyperParametor {
    fn new() -> Self {
        let args = std::env::args().collect::<Vec<String>>();
        HyperParametor {
            time_limit: args
                .get(1)
                .map(|s| s.parse().unwrap())
                .unwrap_or(1.95 * 1000.0),
            start_temp: args.get(2).map(|s| s.parse().unwrap()).unwrap_or(1.0),
            end_temp: args.get(3).map(|s| s.parse().unwrap()).unwrap_or(1e-5),
        }
    }
}

/// 出力する情報
pub struct Output {
    /// d(y, x): 作成した地図のマス(y, x)の色, 0 <= d(y, x) <= m
    d: Vec<Vec<usize>>,
}
impl Output {
    fn write(&self, input: &Input) {
        // dのパディングを除いた部分を出力
        for y in 1..=input.n {
            let mut buf = Vec::with_capacity(input.n);
            for x in 1..=input.n {
                assert!(self.d[y][x] < input.m1);
                buf.push(self.d[y][x]);
            }
            let buf = buf.iter().map(|dyx| dyx.to_string()).join(" ");
            println!("{}", buf);
        }
    }
}

mod solver {
    use super::*;

    /// 解を表現する情報
    #[derive(Clone, Debug)]
    struct State {
        /// d(y, x): 作成した地図のマス(y, x)の色, 0 <= d(y, x) <= m
        d: Grid<usize>,
        /// adj(c1, c2): 区c1とc2が隣接しているかどうか
        adj: Grid<bool>,
        /// score: ZEROの個数
        score: i32,
    }
    impl State {
        fn new(input: &Input) -> Self {
            State {
                d: input.c.clone(),
                adj: input.adj.clone(),
                score: 1,
            }
        }

        /// スコア計算：`Θ(n^2)`
        /// 「作成した地図に含まれる色 0 のマスの総数を E としたとき、E+1 の得点が得られる。」
        /// 差分計算が容易なので，近傍操作中はこの関数を呼ばない．初期化と最後の確認に使う
        fn eval_score(&mut self, input: &Input) -> i32 {
            self.score = 1;
            for y in 1..=input.n {
                for x in 1..=input.n {
                    if self.d[(y, x)] == 0 {
                        self.score += 1;
                    }
                }
            }
            self.score
        }

        /// TODO: 高速化
        /// 制約チェック：`O(n^2 + m^2)`
        /// 隣接関係
        fn is_valid_adj(&mut self, input: &Input) -> bool {
            self.adj = Grid::with_default(input.m1, input.m1);
            for y in 0..input.n2 {
                for x in 0..input.n2 {
                    let c1 = self.d[(y, x)];
                    for (ny, nx) in self.adj.nynx4(y, x) {
                        let c2 = self.d[(ny, nx)];
                        self.adj[(c1, c2)] = true;
                        self.adj[(c2, c1)] = true;
                        if !input.adj[(c1, c2)] {
                            return false;
                        }
                    }
                }
            }
            for c1 in 0..input.m1 {
                for &c2 in input.around[c1].iter() {
                    if !self.adj[(c1, c2)] {
                        return false;
                    }
                }
            }
            true
        }

        /// 制約チェック：`O(1)`
        /// 連結性
        /// 消去可能性を使って，3x3マスだけで判定
        /// ```
        /// 0 1 2
        /// 3 4 5
        /// 6 7 8
        /// ```
        /// の時，4の色c1をc2にしても連結性が変わらない
        /// <=>
        /// 1 + 3 + 5 + 7 - 013 - 215 - 637 - 857 == 1
        fn is_valid_conn_fast(&self, cy: usize, cx: usize) -> bool {
            let c1 = self.d[(cy, cx)];
            // center: d[cy][cx] = a[4]
            let mut a = vec![0_u8; 9];
            let mut local_d = vec![vec![0; 3]; 3];
            for ay in 0..3 {
                for ax in 0..3 {
                    let yy = ay as isize - 1;
                    let xx = ax as isize - 1;
                    let dy = (cy as isize + yy) as usize;
                    let dx = (cx as isize + xx) as usize;
                    let c3 = self.d[(dy, dx)];
                    local_d[ay][ax] = c3;
                    if c1 == c3 {
                        a[ay * 3 + ax] = 1;
                    }
                }
            }
            // eprintln!(
            //     "local_d:\n{:?}\n{:?}\n{:?}",
            //     &local_d[0], &local_d[1], &local_d[2]
            // );
            // eprintln!("a:\n{:?}\n{:?}\n{:?}\n", &a[0..3], &a[3..6], &a[6..9]);
            (a[1] + a[3] + a[5] + a[7]
                - (a[0] * a[1] * a[3])
                - (a[2] * a[1] * a[5])
                - (a[6] * a[3] * a[7])
                - (a[8] * a[5] * a[7]))
                == 1
        }

        /// 近傍操作を適用
        /// now: d(y, x) = c1
        /// next: d(y, x) <- c2
        fn apply_op(&mut self, y: usize, x: usize, c2: usize) {
            self.d[(y, x)] = c2;
        }

        /// 近傍操作を取り消し
        /// now: d(y, x) = c2
        /// next: d(y, x) <- c1
        fn rollback_op(&mut self, y: usize, x: usize, c1: usize) {
            self.d[(y, x)] = c1;
        }
    }

    pub fn solve(input: &Input, hparam: &HyperParametor) -> Output {
        let init_state = State::new(input);
        let mut best_state = annealing(input, hparam, &init_state);
        eprintln!("Score = {}", best_state.eval_score(input));
        Output {
            d: best_state.d.v.clone(),
        }
    }

    /// score_delta, temp の時，近傍操作を受理するかどうか
    /// score_deltaは正の方が良いとする（`max. score`）
    #[inline]
    fn is_accept(rng: &mut rand_pcg::Pcg64Mcg, score_delta: i32, temp: f64) -> bool {
        score_delta > 0 || rng.gen_bool(f64::exp(score_delta as f64 / temp))
    }

    fn annealing(input: &Input, hparam: &HyperParametor, init_state: &State) -> State {
        // parameter
        let mut temp = hparam.start_temp;
        let mut rng = rand_pcg::Pcg64Mcg::new(42);
        // solution
        let mut state = init_state.clone();
        let init_score = state.eval_score(input);
        // info
        const NEIGHBOR_TYPE_NUM: usize = 1;
        let mut iter_cnt = 0u32;
        let mut iter_each_cnt = vec![0; NEIGHBOR_TYPE_NUM];
        let mut move_cnt = vec![0; NEIGHBOR_TYPE_NUM];
        let mut update_cnt = vec![0; NEIGHBOR_TYPE_NUM];

        // ZEROに隣接するマス(!=0)を列挙
        let mut que = std::collections::VecDeque::new();
        for y in 1..=input.n {
            for x in 1..=input.n {
                let c1 = state.d[(y, x)];
                if c1 == 0 {
                    continue;
                }
                for (ny, nx) in state.d.nynx4(y, x) {
                    let c2 = state.d[(ny, nx)];
                    if c2 == 0 {
                        que.push_back((y, x));
                        break;
                    }
                }
            }
        }

        'main: loop {
            iter_cnt += 1;
            // if (iter_cnt & 0b111111) == 0 {
            if (iter_cnt & 0b111) == 0 {
                // 時間更新
                // t: [0, 1]
                let t = input.start_time.elapsed().as_millis() as f64 / hparam.time_limit;
                if t >= 1.0 {
                    break 'main;
                }
                // 温度更新
                // temp: [start_temp, end_temp]
                temp = f64::powf(hparam.start_temp, 1.0 - t) * f64::powf(hparam.end_temp, t);
            }

            // 近傍操作
            let coin = 0;
            iter_each_cnt[coin] += 1;

            // select
            // now: d(y, x) = c1
            // next: d(y, x) <- c2 (c1 != c2)
            let (y, x, c1, c2) = if que.is_empty() {
                // c1 を c2(!=c1) に変える
                'yx: loop {
                    let (y, x) = (rng.gen_range(1..=input.n), rng.gen_range(1..=input.n));
                    let c1 = state.d[(y, x)];
                    let mut around_c = Vec::with_capacity(4);
                    for (ny, nx) in state.d.nynx4(y, x) {
                        let c2 = state.d[(ny, nx)];
                        if c2 != c1 {
                            around_c.push(c2);
                        }
                    }
                    let _ = around_c.iter().sorted().dedup();
                    if around_c.len() > 1 {
                        let c2 = *around_c.first().unwrap();
                        break 'yx (y, x, c1, c2);
                    }
                }
            } else {
                // c1 (!=0) を c2 (=0) に変える
                let (y, x) = que.pop_front().unwrap();
                let c1 = state.d[(y, x)];
                if c1 == 0 {
                    continue 'main;
                }
                let c2 = 0;
                (y, x, c1, c2)
            };

            // score check
            let score_delta = match (c1, c2) {
                (0, _) => -1,
                (_, 0) => 1,
                (_, _) => 0,
            };
            if !is_accept(&mut rng, score_delta, temp) {
                #[cfg(feature = "local")]
                eprintln!("{}|reject", iter_cnt);
                continue 'main;
            }

            // assert_eq!(state.is_valid_conn_naive(input), state.is_valid_conn_fast(y, x));
            // if !state.is_valid_conn_naive(input) {
            if !state.is_valid_conn_fast(y, x) {
                #[cfg(feature = "local")]
                eprintln!("{}|invalid conn|({},{})", iter_cnt, y - 1, x - 1);
                continue 'main;
            }

            let is_alone = {
                let mut f = true;
                for (ny, nx) in state.d.nynx4(y, x) {
                    let c3 = state.d[(ny, nx)];
                    if c3 == c2 {
                        f = false;
                        break;
                    }
                }
                f
            };
            if is_alone {
                #[cfg(feature = "local")]
                eprintln!("{}|alone|({},{})", iter_cnt, y - 1, x - 1);
                continue 'main;
            }

            // apply
            // これ以降に continue する場合は rollback() が必要
            state.apply_op(y, x, c2);

            if !state.is_valid_adj(input) {
                #[cfg(feature = "local")]
                eprintln!("{}|invalid adj|({},{})", iter_cnt, y - 1, x - 1);
                state.rollback_op(y, x, c1);
                continue 'main;
            }

            // update que
            if c2 == 0 {
                for (ny, nx) in state.d.nynx4(y, x) {
                    let c3 = state.d[(ny, nx)];
                    if c3 != 0 {
                        que.push_back((ny, nx));
                    }
                }
            }

            // accept
            state.score += score_delta;
            #[cfg(feature = "local")]
            eprintln!(
                "{}|accept|score+{}={}|{}",
                iter_cnt,
                score_delta,
                state.score,
                match score_delta {
                    1 => "\t\t^^^",
                    -1 => "\t\tvvv",
                    _ => "\t\t---",
                }
            );
            move_cnt[coin] += 1;
            if score_delta > 0 {
                update_cnt[coin] += 1;
                // vis
                #[cfg(feature = "local")]
                if (update_cnt[coin] % 100) == 0 {
                    let out = Output {
                        d: state.d.v.clone(),
                    };
                    out.write(input);
                }
            }
        } // end of 'main

        eprintln!("==================== annealing ====================");
        eprintln!(
            "elapsed\t\t:{} / {} [ms]",
            input.start_time.elapsed().as_millis(),
            hparam.time_limit
        );
        eprintln!("iter\tmove\t:{:?}", move_cnt);
        eprintln!("\tupdate\t:{:?}", update_cnt);
        eprintln!("\ttotal\t:{:?},sum={}", iter_each_cnt, iter_cnt);
        eprintln!("score\tinit\t:{}", init_score);
        eprintln!("\tbest\t:{}", state.eval_score(input));
        eprintln!("===================================================");

        state
    }

    #[cfg(test)]
    mod test {
        use crate::grid;

        #[test]
        fn test_conn_fast() {
            let d = vec![
                vec![0, 0, 0, 0, 0],
                vec![0, 0, 0, 1, 0],
                vec![0, 3, 1, 1, 0],
                vec![0, 2, 2, 2, 0],
                vec![0, 0, 0, 0, 0],
            ];
            let state = super::State {
                d: grid::Grid::new(5, 5, d),
                adj: grid::Grid::with_default(4, 4),
                score: 0,
            };
            assert!(!state.is_valid_conn_fast(2, 1));
            assert!(!state.is_valid_conn_fast(2, 3));
            assert!(state.is_valid_conn_fast(3, 1));
            assert!(!state.is_valid_conn_fast(3, 2));
        }
    }
}

/// # 2次元グリッドライブラリ
///
/// ## 機能
/// - to_index(), from_index()
/// - 座標の4近傍, 8近傍 (範囲外の座標は含まない) を取得
///
/// ## Example
/// ```
/// let v = vec![
///     vec![0, 1, 2],
///     vec![3, 4, 5],
///     vec![6, 7, 8],
/// ];
/// let g = grid::Grid::new(3, 3, v);
/// let (y, x) = (1, 1);
/// for (ny, nx) in g.nynx4(y, x) {
///     // do something
///     eprintln!("(ny,nx)=({},{}), {}", ny, nx, g[(ny, nx)]);
/// }
/// println!("{}", g);
/// eprintln!("{:?}", g);
/// ```
///
/// ## Note
/// 上記の実装だと，for .. in g.nynx(); の中でgに対して操作出来ない．
/// g.nynx() でimmutable borrowが発生しており，immutableなborrow中にmutableなborrow（添え字アクセス）は出来ないため．エラーになる．
/// 競プロやってて初めてRustの所有権で怒られた気がする
/// やっぱり，多くの人がやってるみたいに，for dir .. in D; して，next = cur + dir; is_inside(); するしか無いのか？
/// deltaをforで回すんじゃなく，（範囲チェック済みの）nextをforで回せたら嬉しいなと思ったんだけど無理？もうちょっと考えてみるか
///
mod grid {
    use itertools::Itertools;

    /// Up, Right, Down, Left
    const DYDX4: [(isize, isize); 4] = [(-1, 0), (0, 1), (1, 0), (0, -1)];
    /// Up, UpRight, Right, DownRight, Down, DownLeft, Left, UpLeft
    const DYDX8: [(isize, isize); 8] = [
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
    ];

    // struct DYDX {
    //     dy: isize,
    //     dx: isize,
    // }
    // struct Cell {
    //     y: usize,
    //     x: usize,
    // }
    // impl std::ops::Add<DYDX> for Cell {
    //     type Output = Cell;
    //     /// Cell{y,x} + DYDX{dy,dx} = Cell{y+dy,x+dx}
    //     fn add(self, rhs: DYDX) -> Self::Output {
    //         Cell {
    //             y: (self.y as isize + rhs.dy) as usize,
    //             x: (self.x as isize + rhs.dx) as usize,
    //         }
    //     }
    // }

    #[derive(Clone)]
    pub struct Grid<T> {
        h: usize,
        w: usize,
        pub v: Vec<Vec<T>>,
    }

    impl<T> Grid<T> {
        pub fn new(h: usize, w: usize, v: Vec<Vec<T>>) -> Self {
            Grid { h, w, v }
        }

        pub fn with_default(h: usize, w: usize) -> Self
        where
            T: Default + Clone,
        {
            Grid {
                h,
                w,
                v: vec![vec![T::default(); w]; h],
            }
        }

        /// i: index = y * w + x
        #[inline]
        fn yx_to_index(&self, y: usize, x: usize) -> usize {
            y * self.w + x
        }

        /// i: index
        /// (y, x) = (i / w, i % w)
        #[inline]
        fn index_to_yx(&self, i: usize) -> (usize, usize) {
            (i / self.w, i % self.w)
        }

        /// (y, x)の4近傍の座標を返す
        /// 範囲外の座標は含まない
        // pub fn nynx4(&self, y: usize, x: usize) -> impl Iterator<Item = (usize, usize)> + '_ {
        //     DYDX4
        //         .iter()
        //         .map(move |&(dy, dx)| (y as isize + dy, x as isize + dx))
        //         .filter(|&(ny, nx)| 0 <= ny && 0 <= nx)
        //         .map(|(ny, nx)| (ny as usize, nx as usize))
        //         .filter(|&(ny, nx)| ny < self.h && nx < self.w)
        // }
        pub fn nynx4(&self, y: usize, x: usize) -> Vec<(usize, usize)> {
            DYDX4
                .iter()
                .map(move |&(dy, dx)| (y as isize + dy, x as isize + dx))
                .filter(|&(ny, nx)| 0 <= ny && 0 <= nx)
                .map(|(ny, nx)| (ny as usize, nx as usize))
                .filter(|&(ny, nx)| ny < self.h && nx < self.w)
                .collect_vec()
        }

        /// (y, x)の8近傍の座標を返す
        /// 範囲外の座標は含まない
        pub fn nynx8(&self, y: usize, x: usize) -> impl Iterator<Item = (usize, usize)> + '_ {
            DYDX8
                .iter()
                .map(move |&(dy, dx)| (y as isize + dy, x as isize + dx))
                .filter(|&(ny, nx)| 0 <= ny && 0 <= nx)
                .map(|(ny, nx)| (ny as usize, nx as usize))
                .filter(|&(ny, nx)| ny < self.h && nx < self.w)
        }
    }

    impl<T> std::ops::Index<(usize, usize)> for Grid<T> {
        type Output = T;
        /// `let g = grid::Grid::new(...);`
        /// に対して，
        /// `g.v[y][x]` ではなく，`g[(y, x)]` でアクセスできるようにする
        fn index(&self, (y, x): (usize, usize)) -> &Self::Output {
            &self.v[y][x]
        }
    }

    impl<T> std::ops::IndexMut<(usize, usize)> for Grid<T> {
        /// `let g = grid::Grid::new(...);`
        /// に対して，
        /// `g.v[y][x]` ではなく，`g[(y, x)]` でアクセスできるようにする
        fn index_mut(&mut self, (y, x): (usize, usize)) -> &mut Self::Output {
            &mut self.v[y][x]
        }
    }

    impl<T> std::fmt::Display for Grid<T>
    where
        T: std::fmt::Display,
    {
        /// print!(), println!()
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let buf = self
                .v
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|col| col.to_string())
                        .collect::<Vec<String>>()
                        .join(" ")
                })
                .collect::<Vec<String>>()
                .join("\n");
            write!(f, "{}", buf)?;
            Ok(())
        }
    }

    impl<T> std::fmt::Debug for Grid<T>
    where
        T: std::fmt::Debug,
    {
        /// eprint!(), eprintln!()
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            writeln!(f, "[")?;
            for y in 0..self.h {
                writeln!(f, " {:?}", self.v[y])?;
            }
            write!(f, "]")
        }
    }
}
