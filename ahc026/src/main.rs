use proconio::input;

/// entry point
fn main() {
    let input = Input::new();
    solver::solve(&input);
}

pub struct Input {
    /// n := 箱の個数, n==200
    n: usize,
    /// m := 山の個数, m==10
    m: usize,
    /// b(i,j) := 山iの下からj番目にある箱のid, 1 <= b(i,j) <= n
    b: Vec<Vec<usize>>,
}
impl Input {
    pub fn new() -> Self {
        input! {
            n: usize,
            m: usize,
            b: [[usize; n/m]; m],
        }
        assert!(n == 200);
        assert!(m == 10);
        Input { n, m, b }
    }
}

pub struct Output {
    v: Vec<usize>,
    i: Vec<usize>,
}
impl Output {
    pub fn write(&self) {
        let op_len = self.v.len();
        assert!(op_len <= 5000);
        assert!(self.i.len() == op_len);
        for k in 0..op_len {
            println!("{} {}", self.v[k], self.i[k]);
        }
    }
}

/// ```
/// if i(k) > 0 {
///     箱v(k)とその上に積まれている全ての箱を別の山i(k)に移動
/// } else {
///     箱v(k)を運び出す
/// }
/// ```
#[derive(Clone, Debug)]
pub struct State {
    /// op_v(k) := 箱v
    op_v: Vec<usize>,
    /// op_i(k) := if op_i > 0 { 山i } else { 運び出す }
    op_i: Vec<usize>,
    score: isize,
    used_energy: isize,
    /// stack(i,j) := 山iの下からj番目にある箱のid, 1 <= stack(i,j) <= n
    stack: Vec<Vec<usize>>,
    /// stack_id(v) := 箱vが置いてある山のSome(id), 運び出されている場合はNone
    stack_id: Vec<Option<usize>>,
    /// 次に消す箱のid, 1 <= target_v <= n
    target_v: usize,
}
impl State {
    pub fn new(input: &Input) -> Self {
        let stack = input.b.clone();
        let mut stack_id = vec![Some(0); input.n + 1];
        for i in 0..input.m {
            for j in 0..stack[i].len() {
                stack_id[stack[i][j]] = Some(i);
            }
        }
        State {
            op_v: Vec::with_capacity(input.n),
            op_i: Vec::with_capacity(input.n),
            score: 0_isize,
            used_energy: 0_isize,
            stack,
            stack_id,
            target_v: 1,
        }
    }

    /// * スコア計算
    /// * `O(1)`
    pub fn compute_score(&mut self) -> isize {
        self.score = std::cmp::max(1, 10000 - self.used_energy);
        self.score
    }

    /// * 山iの場所jにある箱v_kとその上を山i_kの一番上に移動
    /// * `O(nj), n==200`
    pub fn move_op(&mut self, v_k: usize, i_k: usize) {
        self.op_v.push(v_k);
        self.op_i.push(i_k + 1); // iは1-indexed
        let (i, j) = self.v_ij(v_k);
        assert!(i != i_k);
        self.used_energy += (self.batch_size(i, j) + 1) as isize;
        for jj in j..self.stack[i].len() {
            let v_i_jj = self.stack[i][jj];
            self.stack_id[v_i_jj] = Some(i_k);
        }
        // stack[i_k][..]+=stack[i][j..]
        let tail = &self.stack[i].clone()[j..];
        self.stack[i_k].extend_from_slice(tail);
        // stack[i] = stack[i][..j]
        self.stack[i].truncate(j);
    }

    /// * 箱vkを運び出す
    /// * `O(1)`
    pub fn carry_out_op(&mut self, v_k: usize) {
        let (i, j) = self.v_ij(v_k);
        assert!(
            self.ij_is_top(i, j),
            "[carryout] j is not top, v_k={}, target_v={}, i={}, j={}, stack[i]={:?}",
            v_k,
            self.target_v,
            i,
            j,
            self.stack[i]
        );
        assert!(self.target_v == v_k);
        self.target_v += 1;
        self.op_v.push(v_k);
        self.op_i.push(0);
        self.stack[i].pop().unwrap();
        self.stack_id[v_k] = None;
    }

    /// * carry_outが出来ないか試す
    /// * `O(n), n==200`
    pub fn try_carry_out(&mut self, input: &Input) {
        while self.target_v <= input.n {
            #[cfg(feature = "local")]
            eprintln!("[try] target_v={}", self.target_v);
            if self.v_is_top(self.target_v) {
                self.carry_out_op(self.target_v);
            } else {
                return;
            }
        }
    }

    /// * 箱vkの場所（山iの下からj番目）
    /// * `O(n), n==200`
    fn v_ij(&self, v_k: usize) -> (usize, usize) {
        if let Some(i) = self.stack_id[v_k] {
            let j = self.stack[i].iter().position(|&v| v == v_k).unwrap();
            (i, j)
        } else {
            (usize::MAX, usize::MAX)
        }
    }

    /// * 山iの下からj番目が一番上かどうか
    /// * `O(1)`
    fn ij_is_top(&self, i: usize, j: usize) -> bool {
        if self.stack[i].is_empty() {
            // 山iが空
            return false;
        }
        j == (self.stack[i].len() - 1)
    }

    /// * 箱v_kが山の一番上かどうか
    /// * `O(1)`
    fn v_is_top(&self, v_k: usize) -> bool {
        if self.stack_id[v_k].is_none() {
            // 既に運び出されている
            return false;
        }
        let (i, j) = self.v_ij(v_k);
        self.ij_is_top(i, j)
    }

    /// * 山iの下からj番目より上にある箱の個数
    /// * `O(1)`
    fn batch_size(&self, i: usize, j: usize) -> usize {
        self.stack[i][j..].len()
    }

    /// * 箱v_kを山iから移す先のid (=i_k) を返す
    fn compute_next_stack_id(&self, input: &Input, v_k: usize) -> usize {
        let (i, _) = self.v_ij(v_k);
        let top_vs = (0..input.m)
            .filter(|&ii| ii != i)
            .filter(|&ii| !self.stack[ii].is_empty())
            .map(|ii| (ii, *self.stack[ii].last().unwrap()))
            .collect::<Vec<_>>();
        // v_kより小さくて，v_kとの差が一番小さい山を探す
        let (i_k, _) = top_vs
            .iter()
            .map(|&(ii, v)| {
                let diff = (v as isize - v_k as isize).abs();
                if v < v_k {
                    (ii, diff)
                } else {
                    (ii, diff * 1000)
                }
            })
            .min_by_key(|&(_, d)| d)
            .unwrap();
        i_k
    }
}

mod solver {
    use super::*;
    use itertools::Itertools;

    pub fn solve(input: &Input) {
        let mut state = State::new(input);
        // TODO: 以下の操作をplayoutして，sortする順番を山登りする
        for target_i in 0..input.m {
            state.try_carry_out(input);
            sort_stack(input, &mut state, target_i);
        }
        while state.target_v <= input.n {
            if !state.v_is_top(state.target_v) {
                // 邪魔なのでどける
                let (i, j) = state.v_ij(state.target_v);
                let v_k1 = state.stack[i][j + 1];
                let i_k = state.compute_next_stack_id(input, v_k1);
                #[cfg(feature = "local")]
                eprintln!("[jyama] j={}, stack={:?}", j, state.stack[i]);
                state.move_op(v_k1, i_k);
            }
            state.carry_out_op(state.target_v);
        }

        let output = Output {
            v: state.op_v.clone(),
            i: state.op_i.clone(),
        };
        output.write();
        eprintln!("Score = {}", state.compute_score());
    }

    /// * 山iをsortする
    fn sort_stack(input: &Input, state: &mut State, target_i: usize) {
        // 山iの箱を，山i以外に退避
        #[cfg(feature = "local")]
        eprintln!("[buf] target_i={}, target_v={}", target_i, state.target_v);
        'buf: while !state.stack[target_i].is_empty() {
            let v_k = *state.stack[target_i].last().unwrap();
            if v_k == state.target_v {
                state.carry_out_op(v_k);
                continue 'buf;
            }
            let i_k = state.compute_next_stack_id(input, v_k);
            state.move_op(v_k, i_k);
        }
        assert!(
            state.stack[target_i].is_empty(),
            "stack[{}]={:?}",
            target_i,
            state.stack[target_i]
        );
        // 山iに戻す
        #[cfg(feature = "local")]
        eprintln!("[reloc] target_i={}, target_v={}", target_i, state.target_v);
        'reloc: loop {
            // 各stackの一番上の箱のidを取得
            // stackが空の行は無視する
            let top_vs = (0..input.m)
                .filter(|&i| i != target_i)
                .filter(|&i| !state.stack[i].is_empty())
                .map(|i| *state.stack[i].last().unwrap())
                .collect::<Vec<_>>();
            if top_vs.is_empty() {
                // 前処理の時点で，全部carry_outできた場合など
                break 'reloc;
            }
            let mut is_success = false;
            'pair: for v_k in top_vs.iter().sorted().rev() {
                let n1 = input.n + 1;
                let base_v = state.stack[target_i].last().unwrap_or(&n1);
                // sortした状態にしたいので，
                // base_v > v_k ならok,
                if *base_v <= *v_k {
                    continue;
                }
                state.move_op(*v_k, target_i);
                is_success = true;
                break 'pair;
            }
            if !is_success {
                break 'reloc;
            }
        }
    }
}
