import sys


class Solver:
    def __init__(self) -> None:
        """
        n = 200, 箱の個数
        m = 10, 山の個数
        B[i][j] := 山iの下からj番目にある箱の番号, \in [1, n], idはunique
        """
        self.n, self.m = map(int, input().split())
        self.B = [list(map(int, input().split())) for _ in range(self.m)]

        self.K = self.n
        # ans = [(v1, i1), (v2, i2), ...]
        # i==0 -> 箱vを運び出す
        # i>0 -> 箱vとその上の箱を山iに移動する
        self.ans: list[tuple[int, int]] = list()

        # 体力消費
        self.used_power = 0
        return

    def write(self) -> None:
        for ans_i in self.ans:
            print(*ans_i)
        return

    def eval_score(self) -> None:
        print("Score =", max(1, 10000 - self.used_power), file=sys.stderr)

    def solve(self) -> None:
        min_v = 1
        for _ in range(self.K):
            y, x = self.find_v_yx(min_v)
            if not self.is_top(y, x):
                # 邪魔なやつをどける
                next_stacks = self.next_stack(y, x)
                BB = [(b, xx) for xx, b in enumerate(self.B[y]) if xx > x + 1]
                cut_lines = [xx for b, xx in sorted(BB, reverse=True)[:len(next_stacks) - 1]]
                cut_lines.append(x + 1)
                cut_lines.sort(reverse=True)
                for yy, c in zip(next_stacks, cut_lines):
                    self.move(self.B[y][c], yy)
            # 運び出す
            self.carry_out(min_v)
            min_v += 1
        return

    def find_v_yx(self, v: int) -> tuple[int, int]:
        for y in range(self.m):
            for x in range(len(self.B[y])):
                if self.B[y][x] == v:
                    return (y, x)
        return None

    def is_top(self, y: int, x: int) -> bool:
        """
        山の頂上かどうか
        """
        return x == len(self.B[y]) - 1

    def stack_size(self, y: int, x: int) -> int:
        """
        山yの[x:]にある箱の個数
        """
        return len(self.B[y][x:])

    def next_stack(self, y: int, x: int) -> bool:
        """
        次の山のid
        """
        # return (y + 1) % self.m
        m = list()
        for yy in range(self.m):
            if yy == y:
                continue
            # 一番小さい箱（次に取り出す）が大きいほうが良い
            # 小さい山に積んでいきたい
            if len(self.B[yy]) > 0:
                cost = - 5 * min(self.B[yy]) + self.stack_size(yy, 0)
            else:
                cost = -float("inf")
            m.append(
                (cost, yy)
            )
        m.sort()
        if self.stack_size(y, x + 1) <= 10:
            return [m[0][-1]]
        else:
            c = min(3, [b > 125 for b in self.B[y][x + 1:]].count(True))
            c = max(1, c)
            return [m_i[-1] for m_i in m[:c]]

    def move(self, vk: int, ik: int) -> None:
        """
        箱vkとその上を山ikに移動する
        """
        self.ans.append((vk, ik + 1))  # 1-indexed
        y, x = self.find_v_yx(vk)
        self.used_power += (len(self.B[y][x:]) + 1)
        self.B[ik] = self.B[ik] + self.B[y][x:]
        self.B[y] = self.B[y][:x]
        return

    def carry_out(self, vk: int) -> None:
        """
        箱vkを運び出す
        """
        self.ans.append((vk, 0))
        y, x = self.find_v_yx(vk)
        assert self.is_top(y, x)
        self.B[y] = self.B[y][:x]
        return


if __name__ == '__main__':
    solver = Solver()
    solver.solve()
    solver.write()
    solver.eval_score()
