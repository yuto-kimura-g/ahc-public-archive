use proconio::input;
use std::time::Instant;

fn main() {
    let input = Input::new();
    solver::solve(&input);
}

/// (msec)
const TIME_LIMIT: f64 = 1950.0;
const MOD: usize = 998_244_353;

pub struct Input {
    /// 盤面の大きさ: n == 9
    n: usize,
    /// スタンプの数: m == 20
    m: usize,
    /// 操作回数の上限: k == 81
    k: usize,
    /// 盤面の初期状態: a(y,x)
    a: Vec<Vec<usize>>,
    /// スタンプの値: s(i,y,x)
    s: Vec<Vec<Vec<usize>>>,
    /// スタンプの大きさ: stamp_size == 3
    stamp_size: usize,
    since: std::time::Instant,
}
impl Input {
    fn new() -> Input {
        let since = Instant::now();
        input! {
            n: usize,
            m: usize,
            k: usize,
            a: [[usize; n]; n],
        }
        let mut s = Vec::with_capacity(m);
        let stamp_size = 3;
        for _ in 0..m {
            input! {
                s_i: [[usize; stamp_size]; stamp_size],
            }
            s.push(s_i);
        }
        assert!(n == 9);
        assert!(m == 20);
        assert!(k == 81);
        Input {
            n,
            m,
            k,
            a,
            s,
            stamp_size,
            since,
        }
    }
}

pub struct Output {
    /// i回目に何番目のスタンプを押すか: m(i)
    m: Vec<usize>,
    /// i番目にスタンプを押す左上位置: pq(i)=(y,x)
    pq: Vec<(usize, usize)>,
}
impl Output {
    fn write(&self, input: &Input) {
        assert!(
            self.m.len() == self.pq.len(),
            "m:{}, pq:{}",
            self.m.len(),
            self.pq.len()
        );
        let l = self.m.len();
        assert!(l <= input.k, "l:{}", l);
        println!("{}", l);
        for i in 0..l {
            let (y, x) = self.pq[i];
            assert!(y <= input.m - 1, "y:{}", y);
            assert!(x <= input.m - 1, "x:{}", x);
            println!("{} {} {}", self.m[i], y, x);
        }
    }
}

mod solver {
    use super::*;
    use rand::prelude::*;

    #[derive(Clone, Debug)]
    struct State {
        b: Vec<Vec<usize>>,
        stamp_id: Vec<usize>,
        stamp_pos: Vec<(usize, usize)>,
        score: isize,
    }
    impl State {
        fn new(input: &Input) -> State {
            State {
                b: input.a.clone(),
                stamp_id: Vec::with_capacity(input.k),
                stamp_pos: Vec::with_capacity(input.k),
                // stamp_id: vec![0; input.k],
                // stamp_pos: vec![(0, 0); input.k],
                score: 0_isize,
            }
        }

        /// スコア計算: `O(n^2), n==9`
        fn compute_score(&mut self, input: &Input) -> isize {
            self.score = 0; // reset
            for i in 0..input.n {
                for j in 0..input.n {
                    self.score += (self.b[i][j] % MOD) as isize;
                }
            }
            self.score
        }

        /// `O(k * stamp_size^2), k==81, stamp_size==3`
        fn playout(&mut self, input: &Input) {
            assert!(self.stamp_id.len() == self.stamp_pos.len());
            self.b = input.a.clone();
            for t in 0..self.stamp_id.len() {
                for i in 0..input.stamp_size {
                    for j in 0..input.stamp_size {
                        let (p, q) = self.stamp_pos[t];
                        let (bi, bj) = (i + p, j + q);
                        self.b[bi][bj] += input.s[self.stamp_id[t]][i][j];
                    }
                }
            }
        }
    }

    pub fn solve(input: &Input) {
        // 初期解
        let mut state = State::new(input);
        // state.playout(input);

        // 改善
        state = optimize(input, &state);

        // 出力
        let output = Output {
            m: state.stamp_id.clone(),
            pq: state.stamp_pos.clone(),
        };
        output.write(input);
        #[cfg(feature = "local")]
        {
            eprintln!("Elapsed = {}[msec]", input.since.elapsed().as_millis());
            eprintln!("Score = {}", state.compute_score(input));
        }
        #[cfg(not(feature = "local"))]
        {
            eprintln!("Score = {}", state.score);
        }
    }

    /// 焼きなましで，温度:`temp`，差分:`delta`の時，その近傍操作を受理するかどうか
    /// * scoreなら，deltaが正だと良い
    /// * costなら，deltaが負だと良い
    fn is_accept(rng: &mut rand_pcg::Pcg64Mcg, temp: f64, delta: isize) -> bool {
        // delta < 0 || rng.gen_bool(f64::exp(-delta as f64 / temp)) // cost
        delta > 0 || rng.gen_bool(f64::exp(delta as f64 / temp)) // score
    }

    /// 焼く
    fn optimize(input: &Input, t0_state: &State) -> State {
        let mut rng = rand_pcg::Pcg64Mcg::new(42);
        let local_since = Instant::now();
        let mut time_rate = 0_f64;
        // const START_TEMP: f64 = 1e-1; // 山登り
        const START_TEMP: f64 = 1e12;
        const END_TEMP: f64 = 1e-1;
        let mut temp = START_TEMP;
        let mut state = t0_state.clone();
        // let pool_size = 1024;
        // let mut state_pool = vec![t0_state.clone(); pool_size];
        let mut best_state = t0_state.clone();
        let t0_score = best_state.compute_score(input);
        let mut iter_cnt = 0;
        const NEIGH_TYPE_NUM: usize = 4;
        let mut move_cnt = [0; NEIGH_TYPE_NUM];
        let mut last_move_iter = [0; NEIGH_TYPE_NUM];
        let mut update_cnt = [0; NEIGH_TYPE_NUM];
        let mut last_update_iter = [0; NEIGH_TYPE_NUM];

        'main: loop {
            iter_cnt += 1;
            // if iter_cnt & 0xf == 0 {
            // 時間更新
            time_rate = (local_since.elapsed().as_millis() as f64) / TIME_LIMIT;
            if time_rate >= 1.0 {
                break 'main;
            }
            // 温度更新
            temp = f64::powf(START_TEMP, 1.0 - time_rate) * f64::powf(END_TEMP, time_rate);
            // }

            // 近傍操作
            let neighbor_type = if state.stamp_id.is_empty() {
                0
            } else if state.stamp_id.len() < input.k {
                match rng.gen_range(0..100) {
                    0..=40 => 0,
                    51..=60 => 1,
                    61..=80 => 2,
                    _ => 3,
                }
            } else {
                match rng.gen_range(0..100) {
                    0..=10 => 1,
                    11..=55 => 2,
                    _ => 3,
                }
            };
            match neighbor_type {
                0 => {
                    // 新しいスタンプを押す
                    let n_id = rng.gen_range(0..input.m);
                    let (np, nq) = (
                        rng.gen_range(0..=input.n - 3),
                        rng.gen_range(0..=input.n - 3),
                    );
                    let mut new_state = state.clone();
                    new_state.stamp_id.push(n_id);
                    new_state.stamp_pos.push((np, nq));
                    new_state.playout(input);
                    let cur_score = state.score;
                    let new_score = new_state.compute_score(input);
                    let score_delta = new_score - cur_score;
                    if !is_accept(&mut rng, temp, score_delta) {
                        // reject
                        continue 'main;
                    }
                    // move
                    state = new_state;
                    move_cnt[neighbor_type] += 1;
                    last_move_iter[neighbor_type] = iter_cnt;
                    if state.score > best_state.score {
                        // update
                        best_state = state.clone();
                        update_cnt[neighbor_type] += 1;
                        last_update_iter[neighbor_type] = iter_cnt;
                        eprintln!(
                            "update|type:{}|iter:{}|temp:{:.1}|score:{}",
                            neighbor_type, iter_cnt, temp, best_state.score
                        );
                    }
                }
                1 => {
                    // t番目を消す
                    let t = rng.gen_range(0..state.stamp_id.len());
                    let mut new_state = state.clone();
                    new_state.stamp_id.remove(t);
                    new_state.stamp_pos.remove(t);
                    new_state.playout(input);
                    let cur_score = state.score;
                    let new_score = new_state.compute_score(input);
                    let score_delta = new_score - cur_score;
                    if !is_accept(&mut rng, temp, score_delta) {
                        // reject
                        continue 'main;
                    }
                    // move
                    state = new_state;
                    move_cnt[neighbor_type] += 1;
                    last_move_iter[neighbor_type] = iter_cnt;
                    if state.score > best_state.score {
                        // update
                        best_state = state.clone();
                        update_cnt[neighbor_type] += 1;
                        last_update_iter[neighbor_type] = iter_cnt;
                        eprintln!(
                            "update|type:{}|iter:{}|temp:{:.1}|score:{}",
                            neighbor_type, iter_cnt, temp, best_state.score
                        );
                    }
                }
                2 => {
                    // t番目のスタンプの場所を (cp, cq) -> (np, nq) に変えてみる
                    let t = rng.gen_range(0..state.stamp_id.len());
                    let (cp, cq) = state.stamp_pos[t];
                    let (np, nq) = (
                        rng.gen_range(0..=input.n - 3),
                        rng.gen_range(0..=input.n - 3),
                    );
                    if cp == np && cq == nq {
                        continue 'main;
                    }
                    let mut new_state = state.clone();
                    new_state.stamp_pos[t] = (np, nq);
                    new_state.playout(input);
                    let cur_score = state.score;
                    let new_score = new_state.compute_score(input);
                    let score_delta = new_score - cur_score;
                    if !is_accept(&mut rng, temp, score_delta) {
                        // reject
                        continue 'main;
                    }
                    // move
                    state = new_state;
                    move_cnt[neighbor_type] += 1;
                    last_move_iter[neighbor_type] = iter_cnt;
                    if state.score > best_state.score {
                        // update
                        best_state = state.clone();
                        update_cnt[neighbor_type] += 1;
                        last_update_iter[neighbor_type] = iter_cnt;
                        eprintln!(
                            "update|type:{}|iter:{}|temp:{:.1}|score:{}",
                            neighbor_type, iter_cnt, temp, best_state.score
                        );
                    }
                }
                3 => {
                    // t番目のスタンプidを c_id -> n_id に変えてみる
                    let t = rng.gen_range(0..state.stamp_id.len());
                    let c_id = state.stamp_id[t];
                    let n_id = rng.gen_range(0..input.m);
                    if c_id == n_id {
                        continue 'main;
                    }
                    let mut new_state = state.clone();
                    new_state.stamp_id[t] = n_id;
                    new_state.playout(input);
                    let cur_score = state.score;
                    let new_score = new_state.compute_score(input);
                    let score_delta = new_score - cur_score;
                    if !is_accept(&mut rng, temp, score_delta) {
                        // reject
                        continue 'main;
                    }
                    // move
                    state = new_state;
                    move_cnt[neighbor_type] += 1;
                    last_move_iter[neighbor_type] = iter_cnt;
                    if state.score > best_state.score {
                        // update
                        best_state = state.clone();
                        update_cnt[neighbor_type] += 1;
                        last_update_iter[neighbor_type] = iter_cnt;
                        eprintln!(
                            "update|type:{}|iter:{}|temp:{:.1}|score:{}",
                            neighbor_type, iter_cnt, temp, best_state.score
                        );
                    }
                }
                _ => unreachable!(),
            }
        } // 'main

        #[cfg(feature = "local")]
        {
            eprintln!("{}", std::iter::repeat(">").take(50).collect::<String>());
            eprintln!("\telapsed\t\t:{}[msec]", local_since.elapsed().as_millis());
            eprintln!(
                "\titer\tmove\t:{:?}[times], last={:?}[iter]",
                move_cnt, last_move_iter
            );
            eprintln!(
                "\t\tupdate\t:{:?}[times], last={:?}[iter]",
                update_cnt, last_update_iter
            );
            eprintln!("\t\ttotal\t:{}[times]", iter_cnt);
            eprintln!("\tscore\tt0\t:{}", t0_score);
            eprintln!("\t\tbest\t:{}", best_state.compute_score(input));
            eprintln!("{}", std::iter::repeat("<").take(50).collect::<String>());
        }

        best_state
    }
}
