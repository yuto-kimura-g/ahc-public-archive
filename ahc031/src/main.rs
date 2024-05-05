// 提出時は警告を無視
// #![allow(unused_variables)]
// #![allow(dead_code)]
// #![allow(unused_assignments)]

use itertools::Itertools;
use proconio::{fastout, input};
use rand::prelude::*;
use rectlib::Rect;
#[allow(unused_imports)]
use std::{
    cmp::{max, min},
    collections::BTreeSet,
    time::Instant,
};

#[fastout]
fn main() {
    let input = Input::new();
    solver::solve(&input);
}

/// 実行時間制限 (msec)
/// 全体
const TIME_LIMIT_GLOBAL: f64 = 2900.0;
/// 初期配置
/// 1日あたり，4 -- 40 (msec)
const TIME_LIMIT_OPT_IDMAP: f64 = 200.0;
/// 広げる
/// 1日あたり，53 -- 530 (msec)
const TIME_LIMIT_OPT_AREASIZE: f64 = 2650.0;
// const TIME_LIMIT_OPT_AREASIZE: f64 = 2650.0 / 2.0;
// const TIME_LIMIT_OPT_AREASIZE: f64 = 2650.0 * 10.0;
/// パーティション
/// 1日あたり，1 -- 10 (msec)
const TIME_LIMIT_OPT_PARTITION: f64 = 50.0;
/// 8近傍
const DYDX8: [(usize, usize); 8] = [
    // Up, Up-Right, Right, Down-Right, Down, Down-Left, Left, Up-Left
    (!0, 0),
    (!0, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, !0),
    (0, !0),
    (!0, !0),
];

pub struct Input {
    /// * グリッドの縦横の大きさW, W=1000
    w: usize,
    /// * 日数D, 5 <= D <= 50
    d: usize,
    /// * 各日の予約数N, 5 <= N <= 50
    n: usize,
    /// * d日目のk番目の予約が希望する面積: a(d,k),
    /// * 1 <= a(d,0) <= ... <= a(d,n-1),
    /// * a(d,0)+...+a(d,n-1) <= W^2
    a: Vec<Vec<usize>>,
    /// * 処理の開始時間
    since: std::time::Instant,
    /// 初期配置の格子点の幅
    inner_width: usize,
    /// 初期配置の格子点の高さ
    inner_height: usize,
}
impl Input {
    fn new() -> Input {
        let since = Instant::now();
        input! {
            w: usize,
            d: usize,
            n: usize,
            a: [[usize; n]; d],
        }
        assert!(w == 1000);
        let inner_width = (n as f64).sqrt().floor() as usize;
        let inner_height = (n as f64 / inner_width as f64).ceil() as usize;
        Input {
            w,
            d,
            n,
            a,
            since,
            inner_width,
            inner_height,
        }
    }
}

pub struct Output {
    /// * d日目のk番目の予約に対して貸し出す長方形: rect(d, k)
    rect: Vec<Vec<Rect>>,
}
impl Output {
    fn write(&self, input: &Input) {
        for d in 0..input.d {
            for k in 0..input.n {
                println!("{}", self.rect[d][k]);
            }
        }
    }
}

mod solver {
    use super::*;

    #[derive(Clone, Debug)]
    struct State {
        rect: Vec<Vec<Rect>>,
        area_cost: isize,
        partition_cost: isize,
        score: isize,
    }
    impl State {
        /// * id_map: 初期配置の写像
        fn new(input: &Input, id_map: Vec<Vec<usize>>) -> State {
            // 初期解
            let dummy_rect = Rect::new(0, 0, 1, 1);
            let mut rect = vec![vec![dummy_rect; input.n]; input.d];
            let rect_size = 1;
            for one_day in 0..input.d {
                for id_k in 0..input.n {
                    let (r, c) = index_to_coord(id_k, input.inner_width);
                    let row = input.w / (input.inner_height + 2) * (r + 1);
                    let col = input.w / (input.inner_width + 2) * (c + 1);
                    let one_rect = Rect::new(
                        row - rect_size,
                        col - rect_size,
                        row + rect_size,
                        col + rect_size,
                    );
                    rect[one_day][id_map[one_day][id_k]] = one_rect;
                }
            }
            State {
                rect,
                area_cost: 0,
                partition_cost: 0,
                score: 0,
            }
        }

        /// * `O(DN)`
        /// * return (area_cost, partition_cost)
        fn compute_cost(&mut self, input: &Input) {
            self.area_cost = 0;
            for one_day in 0..input.d {
                for id_k in 0..input.n {
                    let request = input.a[one_day][id_k];
                    let supply = self.rect[one_day][id_k].areasize();
                    self.area_cost += compute_one_area_cost(request, supply);
                }
            }
            self.partition_cost = 0;
            for one_day in 0..input.d {
                if one_day == 0 {
                    continue;
                }
                for id_k in 0..input.n {
                    self.partition_cost +=
                        compute_one_partition_cost(input, &self.rect[one_day][id_k]);
                }
            }
        }

        /// * `O(1)`
        fn compute_score(&mut self) -> isize {
            self.score = (self.area_cost + self.partition_cost) + 1;
            self.score
        }
    }

    #[inline]
    fn compute_one_area_cost(request: usize, supply: usize) -> isize {
        if request <= supply {
            0_isize
        } else {
            100 * (request - supply) as isize
        }
    }

    #[inline]
    fn compute_one_partition_cost(input: &Input, one_rect: &Rect) -> isize {
        // 本当は，隣接関係とか，前日との繋がりとか考える必要がある
        // 一旦外周だけを考慮した近似値で許してほしい
        let mut cost = 0;
        let (ly, lx, ry, rx) = one_rect.unpack();
        // 外周に接している部分はコストゼロ
        if ly != 0 {
            cost += one_rect.width();
        }
        if lx != 0 {
            cost += one_rect.height();
        }
        if ry != input.w - 1 {
            cost += one_rect.width();
        }
        if rx != input.w - 1 {
            cost += one_rect.height();
        }
        cost as isize
    }

    #[fastout]
    pub fn solve(input: &Input) {
        // 前処理
        let mut id_map = vec![(0..input.n).collect_vec(); input.d];
        for one_day in 0..input.d {
            id_map = optimize_idmap(input, &id_map, one_day);
        }
        // 初期解
        let mut state = State::new(input, id_map);
        state.compute_cost(input);
        state.compute_score();
        #[cfg(feature = "local")]
        {
            eprintln!(
                "Initialize: Elapsed = {}[msec]",
                input.since.elapsed().as_millis()
            );
        }

        // 改善
        for one_day in 0..input.d {
            state = optimize_areasize(input, &state, one_day);
            state = optimize_partition(input, &state, one_day);
        }

        // 出力
        let output = Output {
            rect: state.rect.clone(),
        };
        output.write(input);
        #[cfg(feature = "local")]
        {
            eprintln!(
                "Elapsed = {}[msec] / RTL = {}[msec]",
                input.since.elapsed().as_millis(),
                TIME_LIMIT_GLOBAL
            );
            state.compute_cost(input);
            eprintln!("Score = {}", state.compute_score());
        }
    }

    #[inline]
    fn index_to_coord(index: usize, width: usize) -> (usize, usize) {
        (index / width, index % width)
    }

    #[inline]
    fn coord_to_index(row: usize, col: usize, width: usize) -> usize {
        row * width + col
    }

    /// * 初期配置を良い感じにする
    /// * 焼きなまし
    /// * args:
    ///   * id_map: 初期配置
    ///   * one_day: ある日
    fn optimize_idmap(
        input: &Input,
        t0_id_map: &Vec<Vec<usize>>,
        one_day: usize,
    ) -> Vec<Vec<usize>> {
        // 焼きなましで，温度:`temp`，差分:`delta`の時，その近傍操作を受理するかどうか
        // scoreなら，deltaが正だと良い
        // costなら，deltaが負だと良い
        let is_accept = |rng: &mut rand_pcg::Pcg64Mcg, temp: f64, score_delta: f64| -> bool {
            score_delta > 0.0 || rng.gen::<f64>() < f64::exp(score_delta / temp)
        };

        let compute_score = |id_map: &Vec<Vec<usize>>| -> f64 {
            // 8近傍で隣り合うaの差の絶対値の和の平均 -> max
            let mut score = 0.0;
            #[allow(unused_assignments)]
            let mut each_score = 0;
            #[allow(unused_assignments)]
            let mut around_cnt = 0;
            for id_k3 in 0..input.n {
                let (r3, c3) = index_to_coord(id_k3, input.inner_width);
                each_score = 0;
                around_cnt = 0;
                for &(dy, dx) in DYDX8.iter() {
                    let r4 = r3.wrapping_add(dy);
                    let c4 = c3.wrapping_add(dx);
                    if r4 >= input.inner_height || c4 >= input.inner_width {
                        continue;
                    }
                    let id_k4 = coord_to_index(r4, c4, input.inner_width);
                    if id_k4 >= input.n {
                        continue;
                    }
                    each_score += (input.a[one_day][id_map[one_day][id_k3]] as isize
                        - input.a[one_day][id_map[one_day][id_k4]] as isize)
                        .pow(2);
                    around_cnt += 1;
                }
                if around_cnt > 0 {
                    score += each_score as f64 / around_cnt as f64;
                }
            }
            score
        };

        let mut rng = rand_pcg::Pcg64Mcg::new(42);
        let local_since = Instant::now();
        let local_timelimit = TIME_LIMIT_OPT_IDMAP / input.d as f64;
        #[allow(unused_assignments)]
        let mut time_rate = 0_f64;
        const START_TEMP: f64 = 1e1;
        const END_TEMP: f64 = 1e-1;
        #[allow(unused_assignments)]
        let mut temp = START_TEMP;
        let mut iter_cnt = 0;
        let mut move_cnt = 0;
        let mut last_move_iter = 0;
        let mut update_cnt = 0;
        let mut last_update_iter = 0;
        let t0_score = compute_score(t0_id_map);
        let mut id_map = t0_id_map.clone();
        let mut best_id_map = t0_id_map.clone();
        'main: loop {
            iter_cnt += 1;

            time_rate = local_since.elapsed().as_millis() as f64 / local_timelimit;
            if time_rate >= 1.0 {
                break 'main;
            }
            temp = f64::powf(START_TEMP, 1.0 - time_rate) * f64::powf(END_TEMP, time_rate);

            // 近傍操作：2点swap
            let id_k1 = rng.gen_range(0..input.n);
            let id_k2 = rng.gen_range(0..input.n);
            if id_k1 == id_k2 {
                continue 'main;
            }
            let cur_score = compute_score(&id_map);
            id_map[one_day].swap(id_k1, id_k2);
            let new_score = compute_score(&id_map);
            let score_delta = new_score - cur_score;
            if !is_accept(&mut rng, temp, score_delta) {
                // reject, rollback
                id_map[one_day].swap(id_k1, id_k2);
                continue 'main;
            }
            move_cnt += 1;
            last_move_iter = iter_cnt;
            if score_delta > 0.0 {
                update_cnt += 1;
                last_update_iter = iter_cnt;
                best_id_map = id_map.clone();
            }
        }
        #[cfg(feature = "local")]
        {
            eprintln!("=====>>> opt idmap @ day {}", one_day);
            eprintln!("\telapsed\t\t:{}[msec]", local_since.elapsed().as_millis());
            eprintln!(
                "\titer\tmove\t:{}[times], last={}[iter]",
                move_cnt, last_move_iter
            );
            eprintln!(
                "\t\tupdate\t:{}[times], last={}[iter]",
                update_cnt, last_update_iter
            );
            eprintln!("\t\ttotal\t:{}[times]", iter_cnt);
            eprintln!("\tt0_score:\t:{}", (t0_score));
            eprintln!("\tlast_score:\t:{}", (compute_score(&best_id_map)));
            eprintln!("=====<<<");
        }
        best_id_map
    }

    /// * 指定の面積をできるだけ満たすように広げる
    /// * 焼きなまし
    /// * args:
    ///   * t0_state: 初期状態
    ///   * one_day: ある日
    fn optimize_areasize(input: &Input, t0_state: &State, one_day: usize) -> State {
        // 焼きなましで，温度:`temp`，差分:`delta`の時，その近傍操作を受理するかどうか
        // scoreなら，deltaが正だと良い
        // costなら，deltaが負だと良い
        let is_accept = |rng: &mut rand_pcg::Pcg64Mcg, temp: f64, delta: isize| -> bool {
            // delta > 0 || rng.gen_bool(f64::exp(delta as f64 / temp)) // score
            delta < 0 || rng.gen_bool(f64::exp(-delta as f64 / temp)) // cost
        };

        let mut rng = rand_pcg::Pcg64Mcg::new(42);
        let local_since = Instant::now();
        let local_timelimit = TIME_LIMIT_OPT_AREASIZE / input.d as f64;
        let mut time_rate = 0_f64;
        // w * w = 1000 * 1000 = 1e3 * 1e3 = 1e6
        const SCALE_DELTA_MAX: f64 = 30.0;
        const SCALE_DELTA_MIN: f64 = 1.0;
        const SLIDE_DELTA_MAX: f64 = 50.0;
        const SLIDE_DELTA_MIN: f64 = 1.0;
        const START_TEMP: f64 = 1e4;
        const END_TEMP: f64 = 1e-1;
        let mut temp = START_TEMP;
        let mut state = t0_state.clone();
        let t0_score = state.compute_score();
        let mut best_state = state.clone();
        let mut iter_cnt = 0;
        let mut move_cnt = [0; 2];
        let mut last_move_iter = [0; 2];
        let mut update_cnt = [0; 2];
        let mut last_update_iter = [0; 2];

        'main: loop {
            iter_cnt += 1;
            if iter_cnt & 0xf == 0 {
                // 時間更新
                time_rate = local_since.elapsed().as_millis() as f64 / local_timelimit;
                if time_rate >= 1.0 {
                    break 'main;
                }
                // 温度更新
                temp = f64::powf(START_TEMP, 1.0 - time_rate) * f64::powf(END_TEMP, time_rate);
            }

            // 近傍操作
            match rng.gen_range(0..100) {
                0..=5 => {
                    // 近傍１：2点swap
                    let id_k1 = rng.gen_range(0..input.n - 2);
                    let id_k2 = rng.gen_range(id_k1 + 1..input.n);
                    // k1 - cur
                    let k1_req = input.a[one_day][id_k1];
                    let cur_k1_sup = state.rect[one_day][id_k1].areasize();
                    let cur_k1_area_cost = compute_one_area_cost(k1_req, cur_k1_sup);
                    let cur_k1_p_cost =
                        compute_one_partition_cost(input, &state.rect[one_day][id_k1]);
                    // k2 - cur
                    let k2_req = input.a[one_day][id_k2];
                    let cur_k2_sup = state.rect[one_day][id_k2].areasize();
                    let cur_k2_area_cost = compute_one_area_cost(k2_req, cur_k2_sup);
                    let cur_k2_p_cost =
                        compute_one_partition_cost(input, &state.rect[one_day][id_k2]);
                    // k1 - new
                    let new_k1_sup = state.rect[one_day][id_k2].areasize();
                    let new_k1_area_cost = compute_one_area_cost(k1_req, new_k1_sup);
                    let new_k1_p_cost =
                        compute_one_partition_cost(input, &state.rect[one_day][id_k2]);
                    // k2 - new
                    let new_k2_sup = state.rect[one_day][id_k1].areasize();
                    let new_k2_area_cost = compute_one_area_cost(k2_req, new_k2_sup);
                    let new_k2_p_cost =
                        compute_one_partition_cost(input, &state.rect[one_day][id_k1]);
                    // delta
                    let area_cost_delta = (new_k1_area_cost + new_k2_area_cost)
                        - (cur_k1_area_cost + cur_k2_area_cost);
                    let p_cost_delta =
                        (new_k1_p_cost + new_k2_p_cost) - (cur_k1_p_cost + cur_k2_p_cost);
                    if !is_accept(&mut rng, temp, area_cost_delta / 100) {
                        // reject
                        continue 'main;
                    }
                    // 近傍に移動してみる
                    state.rect[one_day].swap(id_k1, id_k2);
                    state.area_cost += area_cost_delta;
                    state.partition_cost += p_cost_delta;
                    move_cnt[0] += 1;
                    last_move_iter[0] = iter_cnt;
                    if state.area_cost < best_state.area_cost {
                        // ベストスコアを更新した
                        best_state = state.clone();
                        update_cnt[0] += 1;
                        last_update_iter[0] = iter_cnt;
                    }
                }
                _ => {
                    // 近傍２：idをランダムに一つ選んで，上下左右にランダムに広げる／縮める／スライドする
                    let id_k = rng.gen_range(0..input.n);
                    let neighbor_type = rng.gen_range(0..12);
                    let (dy, dx) = match neighbor_type {
                        0..=7 => (
                            // Scale
                            rng.gen_range(
                                1..=(SCALE_DELTA_MAX * (1.0 - time_rate)
                                    + SCALE_DELTA_MIN * time_rate)
                                    as usize,
                            ),
                            rng.gen_range(
                                1..=(SCALE_DELTA_MAX * (1.0 - time_rate)
                                    + SCALE_DELTA_MIN * time_rate)
                                    as usize,
                            ),
                        ),
                        8..=11 => (
                            // Slide
                            rng.gen_range(
                                1..=(SLIDE_DELTA_MAX * (1.0 - time_rate)
                                    + SLIDE_DELTA_MIN * time_rate)
                                    as usize,
                            ),
                            rng.gen_range(
                                1..=(SLIDE_DELTA_MAX * (1.0 - time_rate)
                                    + SLIDE_DELTA_MIN * time_rate)
                                    as usize,
                            ),
                        ),
                        _ => unreachable!(),
                    };
                    let (c_ly, c_lx, c_ry, c_rx) = state.rect[one_day][id_k].unpack();
                    let (n_ly, n_lx, n_ry, n_rx) = match neighbor_type {
                        // 改善（拡大）
                        0 => {
                            // up
                            (c_ly.wrapping_sub(dy), c_lx, c_ry, c_rx)
                        }
                        1 => {
                            // down
                            (c_ly, c_lx, c_ry.wrapping_add(dy), c_rx)
                        }
                        2 => {
                            // left
                            (c_ly, c_lx.wrapping_sub(dx), c_ry, c_rx)
                        }
                        3 => {
                            // right
                            (c_ly, c_lx, c_ry, c_rx.wrapping_add(dx))
                        }
                        // 改悪（縮小）
                        4 => {
                            // up
                            (c_ly.wrapping_add(dy), c_lx, c_ry, c_rx)
                        }
                        5 => {
                            // down
                            (c_ly, c_lx, c_ry.wrapping_sub(dy), c_rx)
                        }
                        6 => {
                            // left
                            (c_ly, c_lx.wrapping_add(dx), c_ry, c_rx)
                        }
                        7 => {
                            // right
                            (c_ly, c_lx, c_ry, c_rx.wrapping_sub(dx))
                        }
                        // 維持または悪化（スライド）
                        // 既に端にある場合は，悪化する
                        8 => {
                            // up+down (up)
                            (c_ly.wrapping_sub(dy), c_lx, c_ry.wrapping_sub(dy), c_rx)
                        }
                        9 => {
                            // up+down (down)
                            (c_ly.wrapping_add(dy), c_lx, c_ry.wrapping_add(dy), c_rx)
                        }
                        10 => {
                            // left+right (left)
                            (c_ly, c_lx.wrapping_sub(dx), c_ry, c_rx.wrapping_sub(dx))
                        }
                        11 => {
                            // left+right (right)
                            (c_ly, c_lx.wrapping_add(dx), c_ry, c_rx.wrapping_add(dx))
                        }
                        _ => unreachable!(),
                    };
                    if n_ly >= n_ry || n_lx >= n_rx || n_ry >= input.w || n_rx >= input.w {
                        // 範囲外，制約違反
                        continue 'main;
                    }
                    let new_rect = Rect::new(n_ly, n_lx, n_ry, n_rx);
                    let req = input.a[one_day][id_k];
                    let cur_sup = state.rect[one_day][id_k].areasize();
                    let cur_area_cost = compute_one_area_cost(req, cur_sup);
                    let cur_p_cost = compute_one_partition_cost(input, &state.rect[one_day][id_k]);
                    let new_sup = new_rect.areasize();
                    let new_area_cost = compute_one_area_cost(req, new_sup);
                    let new_p_cost = compute_one_partition_cost(input, &new_rect);
                    let area_cost_delta = new_area_cost - cur_area_cost;
                    let p_cost_delta = new_p_cost - cur_p_cost;
                    if !is_accept(&mut rng, temp, area_cost_delta / 100) {
                        // reject
                        continue 'main;
                    }
                    for id_kk in 0..input.n {
                        if id_k == id_kk {
                            continue;
                        }
                        if state.rect[one_day][id_kk].is_overlap(&new_rect) {
                            continue 'main;
                        }
                    }
                    // 近傍に移動してみる
                    state.rect[one_day][id_k] = new_rect;
                    state.area_cost += area_cost_delta;
                    state.partition_cost += p_cost_delta;
                    move_cnt[1] += 1;
                    last_move_iter[1] = iter_cnt;
                    if state.area_cost < best_state.area_cost {
                        // ベストスコアを更新した
                        best_state = state.clone();
                        update_cnt[1] += 1;
                        last_update_iter[1] = iter_cnt;
                    }
                }
            }
        } // 'main

        #[cfg(feature = "local")]
        {
            eprintln!("=====>>> opt areasize @ day {}", one_day);
            eprintln!("\telapsed\t\t:{}[msec]", local_since.elapsed().as_millis());
            eprintln!("\ttime limit\t:{}[msec]", local_timelimit);
            eprintln!(
                "\titer\tmove\t:{:?}[times], last={:?}[iter]",
                move_cnt, last_move_iter
            );
            eprintln!(
                "\t\tupdate\t:{:?}[times], last={:?}[iter]",
                update_cnt, last_update_iter
            );
            eprintln!("\t\ttotal\t:{}[times]", iter_cnt);
            best_state.compute_cost(input);
            eprintln!(
                "\tcost\tend\t:area{}+p{}",
                best_state.area_cost, best_state.partition_cost
            );
            eprintln!("\tscore\tbegin\t:{}", t0_score);
            eprintln!("\t\tend\t:{}", best_state.compute_score());
            eprintln!("=====<<<");
        }

        best_state
    }

    /// * パーティションをいい感じにする
    /// * 山登り
    /// * args:
    ///   * t0_state: 初期状態
    ///   * one_day: ある日
    fn optimize_partition(input: &Input, t0_state: &State, one_day: usize) -> State {
        let local_since = Instant::now();
        let local_timelimit = TIME_LIMIT_OPT_PARTITION / input.d as f64;
        let mut state = t0_state.clone();
        let t0_score = state.compute_score();
        let mut update_cnt = 0;

        'main: for id_k in (0..input.n).rev() {
            // ギリギリまで広げて隣と共有する／外周と一致させる ことで，パーティションを節約したい
            let (c_ly, c_lx, c_ry, c_rx) = state.rect[one_day][id_k].unpack();
            let (mut n_ly, mut n_lx, mut n_ry, mut n_rx) = state.rect[one_day][id_k].unpack();

            // 上に広げる
            'ly: while n_ly > 0 {
                // 少し広げてみる
                n_ly -= 1;
                let n_rect = Rect::new(n_ly, n_lx, n_ry, n_rx);
                for id_k2 in 0..input.n {
                    if id_k2 == id_k {
                        continue;
                    }
                    let o_rect = &state.rect[one_day][id_k2];
                    if n_rect.is_overlap(o_rect) || o_rect.is_overlap(&n_rect) {
                        // これ以上この方向に広げられない
                        n_ly += 1; // reset
                        break 'ly;
                    }
                }
                if n_ly == 0 {
                    break 'ly;
                }
            }

            // 下に広げる
            'ry: while n_ry < input.w - 1 {
                n_ry += 1;
                let n_rect = Rect::new(n_ly, n_lx, n_ry, n_rx);
                for id_k2 in 0..input.n {
                    if id_k2 == id_k {
                        continue;
                    }
                    let o_rect = &state.rect[one_day][id_k2];
                    if n_rect.is_overlap(o_rect) || o_rect.is_overlap(&n_rect) {
                        n_ry -= 1;
                        break 'ry;
                    }
                }
                if n_ry == input.w - 1 {
                    break 'ry;
                }
            }

            // 左に広げる
            'lx: while n_lx > 0 {
                n_lx -= 1;
                let n_rect = Rect::new(n_ly, n_lx, n_ry, n_rx);
                for id_k2 in 0..input.n {
                    if id_k2 == id_k {
                        continue;
                    }
                    let o_rect = &state.rect[one_day][id_k2];
                    if n_rect.is_overlap(o_rect) || o_rect.is_overlap(&n_rect) {
                        n_lx += 1;
                        break 'lx;
                    }
                }
                if n_lx == 0 {
                    break 'lx;
                }
            }

            // 右に広げる
            'rx: while n_rx < input.w - 1 {
                n_rx += 1;
                let n_rect = Rect::new(n_ly, n_lx, n_ry, n_rx);
                for id_k2 in 0..input.n {
                    if id_k2 == id_k {
                        continue;
                    }
                    let o_rect = &state.rect[one_day][id_k2];
                    if n_rect.is_overlap(o_rect) || o_rect.is_overlap(&n_rect) {
                        n_rx -= 1;
                        break 'rx;
                    }
                }
                if n_rx == input.w - 1 {
                    break 'rx;
                }
            }

            if (c_ly, c_lx, c_ry, c_rx) == (n_ly, n_lx, n_ry, n_rx) {
                // 何も変わらなかった
                continue 'main;
            }
            let new_rect = Rect::new(n_ly, n_lx, n_ry, n_rx);
            #[cfg(feature = "local")]
            {
                if new_rect.areasize() < state.rect[one_day][id_k].areasize() {
                    // 悪化した（そんなはずは無いんですが...）
                    unreachable!(
                        "why???\n{}({}) -> {}({})",
                        state.rect[one_day][id_k],
                        state.rect[one_day][id_k].areasize(),
                        new_rect,
                        new_rect.areasize()
                    );
                }
            }
            let req = input.a[one_day][id_k];
            let cur_sup = state.rect[one_day][id_k].areasize();
            let cur_area_cost = compute_one_area_cost(req, cur_sup);
            let cur_p_cost = compute_one_partition_cost(input, &state.rect[one_day][id_k]);
            let new_sup = new_rect.areasize();
            let new_area_cost = compute_one_area_cost(req, new_sup);
            let new_p_cost = compute_one_partition_cost(input, &new_rect);
            let area_cost_delta = new_area_cost - cur_area_cost;
            let p_cost_delta = new_p_cost - cur_p_cost;
            // 近傍に移動
            state.rect[one_day][id_k] = new_rect;
            state.area_cost += area_cost_delta;
            state.partition_cost += p_cost_delta;
            update_cnt += 1;
        } // 'main

        #[cfg(feature = "local")]
        {
            eprintln!("=====>>> opt partition @ day {}", one_day);
            eprintln!("\telapsed\t\t:{}[msec]", local_since.elapsed().as_millis());
            eprintln!("\ttime limit\t:{}[msec]", local_timelimit);
            eprintln!("\titer\tupdate\t:{}[times] / {}[n]", update_cnt, input.n);
            state.compute_cost(input);
            eprintln!(
                "\tcost\tend\t:area{}+p{}",
                state.area_cost, state.partition_cost
            );
            eprintln!("\tscore\tbegin\t:{}", t0_score);
            eprintln!("\t\tend\t:{}", state.compute_score());
            eprintln!("=====<<<");
        }

        state
    }
}

mod rectlib {
    #[derive(Debug, Clone)]
    pub struct Rect {
        ly: usize,
        lx: usize,
        ry: usize,
        rx: usize,
    }

    impl Rect {
        pub fn new(ly: usize, lx: usize, ry: usize, rx: usize) -> Rect {
            assert!(lx < rx && ly < ry, "l({}, {}),r({}, {})", ly, lx, ry, rx);
            Rect { ly, lx, ry, rx }
        }

        /// * (ly, lx, ry, rx)
        #[inline]
        pub fn unpack(&self) -> (usize, usize, usize, usize) {
            (self.ly, self.lx, self.ry, self.rx)
        }

        #[inline]
        pub fn width(&self) -> usize {
            self.rx - self.lx
        }

        #[inline]
        pub fn height(&self) -> usize {
            self.ry - self.ly
        }

        #[inline]
        pub fn areasize(&self) -> usize {
            self.width() * self.height()
        }

        /// 線で接しているのはOK (`!is_overlap`)
        pub fn is_overlap(&self, other: &Rect) -> bool {
            std::cmp::max(self.ly, other.ly) < std::cmp::min(self.ry, other.ry)
                && std::cmp::max(self.lx, other.lx) < std::cmp::min(self.rx, other.rx)
        }
    }

    impl std::fmt::Display for Rect {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            write!(f, "{} {} {} {}", self.ly, self.lx, self.ry, self.rx)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_rect() {
            let rect_1 = Rect::new(1, 1, 2, 3);
            let rect_2 = Rect::new(0, 0, 1, 1);
            let rect_3 = Rect::new(0, 0, 2, 2);
            assert!(!rect_1.is_overlap(&rect_2));
            assert!(!rect_2.is_overlap(&rect_1));
            assert!(rect_1.is_overlap(&rect_3));
            assert!(rect_3.is_overlap(&rect_1));
        }
    }
}
