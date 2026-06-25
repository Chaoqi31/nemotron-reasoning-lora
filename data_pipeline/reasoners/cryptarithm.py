# Author: Huikang Tong (https://github.com/tonghuikang/nemotron)
"""Cryptarithm solver — concat-first + targeted arithmetic search.

Phase 1: Fast concat/reverse-concat detection.
Phase 2: Try operations greedily. For each arrangement, try the most likely
         operation combo first. Use brute-force for ≤8 symbols, backtracking
         for larger sets. Short per-problem timeout.
"""

from __future__ import annotations

import time as _time
from itertools import permutations as _iperms
from typing import Optional

from reasoners.store_types import Problem


def _compute(op_name: str, left: int, right: int) -> int:
    if op_name == "mul":
        return left * right
    if op_name == "add":
        return left + right
    if op_name == "sub":
        return left - right
    if op_name == "absdiff":
        return abs(left - right)
    if op_name == "cat":
        rlen = len(str(abs(right))) if right != 0 else 1
        return left * (10**rlen) + right
    if op_name == "revcat":
        llen = len(str(abs(left))) if left != 0 else 1
        return right * (10**llen) + left
    return 0


def _result_digits(op_name: str, left: int, right: int) -> str:
    result = _compute(op_name, left, right)
    if result < 0:
        return "-" + str(-result)
    return str(result)


def _make_operands(a_d: int, b_d: int, c_d: int, d_d: int, arr_name: str) -> tuple[int, int]:
    if arr_name == "BA,DC":
        return b_d * 10 + a_d, d_d * 10 + c_d
    if arr_name == "AB,DC":
        return a_d * 10 + b_d, d_d * 10 + c_d
    if arr_name == "BA,CD":
        return b_d * 10 + a_d, c_d * 10 + d_d
    return a_d * 10 + b_d, c_d * 10 + d_d


def _best_ops(out_lengths: list[int]) -> list[str]:
    """Return operations ordered by likelihood for given output lengths.

    For 2-digit operands (10-99):
      mul:     100-9801   → 3-4 digits
      add:     20-198     → 2-3 digits
      sub:     -89..89    → up to 3 chars (with minus)
      absdiff: 0-89       → 1-2 digits
    """
    mn, mx = min(out_lengths), max(out_lengths)
    if mn >= 4:
        # All outputs 4+ digits: ONLY mul can do this
        return ["mul"]
    if mx >= 4:
        # Some 4-digit outputs: mul for sure, maybe add for shorter ones
        return ["mul", "add", "sub", "absdiff"]
    if mn >= 3:
        # All 3+ digits: mul or add (sub/absdiff max 2 digits positive)
        return ["mul", "add", "absdiff", "sub"]
    if mx <= 2:
        # All 2-digit outputs: absdiff or sub (mul min is 100=3 digits)
        return ["absdiff", "sub", "add", "mul"]
    if mn >= 2:
        # 2-3 digit outputs: add is most likely
        return ["add", "mul", "sub", "absdiff"]
    return ["absdiff", "sub", "add", "mul"]


def _verify_all(ex_data, op_assignments, arr_name, assignment):
    for A, B, op, C, D, out_str in ex_data:
        left, right = _make_operands(assignment[A], assignment[B], assignment[C], assignment[D], arr_name)
        op_name = op_assignments.get(op, "cat")
        result = _result_digits(op_name, left, right)
        if not all(ch in assignment for ch in out_str if ch != "-"):
            return False
        expected = 0
        neg = out_str.startswith("-")
        for ch in (out_str[1:] if neg else out_str):
            expected = expected * 10 + assignment[ch]
        if neg:
            expected = -expected
        try:
            if int(result) != expected:
                return False
        except ValueError:
            return False
    return True


def _try_solve(examples, question):
    ex_data = []
    op_chars: set[str] = set()
    for ex in examples:
        inp = str(ex.input_value)
        out = str(ex.output_value)
        if len(inp) < 5:
            return None
        A, B, op, C, D = inp[0], inp[1], inp[2], inp[3], inp[4]
        op_chars.add(op)
        ex_data.append((A, B, op, C, D, out))
    q = question
    if len(q) < 5:
        return None
    qA, qB, qOp, qC, qD = q[0], q[1], q[2], q[3], q[4]
    op_chars.add(qOp)

    digit_syms: set[str] = set()
    for A, B, op, C, D, out_str in ex_data:
        for ch in [A, B, C, D]:
            if ch not in op_chars:
                digit_syms.add(ch)
        for ch in out_str:
            if ch not in op_chars:
                digit_syms.add(ch)
    for ch in [qA, qB, qC, qD]:
        if ch not in op_chars:
            digit_syms.add(ch)
    syms = sorted(digit_syms)
    n_syms = len(syms)
    if n_syms > 10:
        return None

    by_op: dict[str, list] = {}
    for A, B, op, C, D, out_str in ex_data:
        by_op.setdefault(op, []).append((A, B, C, D, out_str))

    # Phase 1: concat detection
    known_ops: dict[str, Optional[str]] = {}
    unknown_ops: list[str] = []
    for op_char, ops in by_op.items():
        all_fwd = all(A + B + C + D == out for A, B, C, D, out in ops)
        all_rev = all(C + D + A + B == out for A, B, C, D, out in ops)
        if all_fwd:
            known_ops[op_char] = "cat"
        elif all_rev:
            known_ops[op_char] = "revcat"
        else:
            known_ops[op_char] = None
            unknown_ops.append(op_char)

    if qOp not in by_op:
        known_ops[qOp] = None
        unknown_ops.append(qOp)

    if not unknown_ops and known_ops.get(qOp):
        op_name = known_ops[qOp]
        assert op_name is not None
        answer = qA + qB + qC + qD if op_name == "cat" else qC + qD + qA + qB
        return (op_name, "AB,CD", answer, {})
    if not unknown_ops:
        return None

    # Phase 2: try operations
    # Get best operation order for each unknown op
    op_orders: dict[str, list[str]] = {}
    for op_char in unknown_ops:
        if op_char in by_op:
            lengths = [len(out) for _, _, _, _, out in by_op[op_char]]
            op_orders[op_char] = _best_ops(lengths)
        else:
            op_orders[op_char] = ["mul", "add", "sub", "absdiff"]

    # Build combo list: most likely first
    # For each op, take the first candidate. Then try alternatives one at a time.
    op_list = list(op_orders.items())
    combos = _generate_combos(op_list)

    arrangements = ["AB,CD", "BA,DC", "AB,DC", "BA,CD"]

    # Total time budget per problem
    if n_syms <= 7:
        budget = 15.0
    elif n_syms <= 8:
        budget = 20.0
    elif n_syms <= 9:
        budget = 5.0
    else:
        budget = 3.0
    start = _time.perf_counter()

    for arr_name in arrangements:
        for combo in combos:
            if _time.perf_counter() - start > budget:
                return None
            op_assignments: dict[str, str] = {}
            for k, v in known_ops.items():
                if v is not None:
                    op_assignments[k] = v
            for op_char, op_name in zip(unknown_ops, combo):
                op_assignments[op_char] = op_name

            remaining = budget - (_time.perf_counter() - start)
            if remaining < 0.1:
                return None

            mapping = _solve_digits(syms, n_syms, ex_data, op_assignments, arr_name, unknown_ops, remaining * 0.8)
            if mapping is not None:
                return _compute_answer(qA, qB, qOp, qC, qD, op_assignments, arr_name, mapping)
    return None


def _generate_combos(op_list):
    """Generate operation combos in priority order (most likely first).

    Strategy:
    1. Best combo (first candidate for each op)
    2. One alternative at a time
    3. Cross-combos (2 alternatives at once) — catches cases where best guess
       for both operators is wrong simultaneously
    """
    if not op_list:
        return [()]
    # Start with the best combo (first candidate for each op)
    best = tuple(candidates[0] for _, candidates in op_list)
    combos = [best]
    seen = {best}
    # Then try one alternative at a time
    for i, (op_char, candidates) in enumerate(op_list):
        for alt in candidates[1:]:
            combo = list(best)
            combo[i] = alt
            t = tuple(combo)
            if t not in seen:
                combos.append(t)
                seen.add(t)
    # Then try cross-combos (2 alternatives at once)
    if len(op_list) >= 2:
        for i, (_, cands_i) in enumerate(op_list):
            for j, (_, cands_j) in enumerate(op_list):
                if j <= i:
                    continue
                for alt_i in cands_i[1:]:
                    for alt_j in cands_j[1:]:
                        combo = list(best)
                        combo[i] = alt_i
                        combo[j] = alt_j
                        t = tuple(combo)
                        if t not in seen:
                            combos.append(t)
                            seen.add(t)
    return combos


def _solve_digits(syms, n_syms, ex_data, op_assignments, arr_name, unknown_ops, timeout):
    start = _time.perf_counter()

    if n_syms <= 8:
        for perm in _iperms(range(10), n_syms):
            if _time.perf_counter() - start > timeout:
                return None
            assignment = dict(zip(syms, perm))
            if _verify_all(ex_data, op_assignments, arr_name, assignment):
                return assignment
        return None
    else:
        # Backtracking with forward checking
        sym_weight: dict[str, int] = {}
        for s in syms:
            w = 0
            for A, B, op, C, D, out_str in ex_data:
                if s in (A, B, C, D):
                    w += 2
                if s in out_str and op in unknown_ops:
                    w += 1
            sym_weight[s] = w
        syms_ordered = sorted(syms, key=lambda s: -sym_weight[s])

        used = [False] * 10
        assignment: dict[str, int] = {}
        iters = [0]

        def forward_check():
            for A, B, op, C, D, out_str in ex_data:
                if op not in unknown_ops:
                    continue
                if not all(s in assignment for s in (A, B, C, D)):
                    continue
                left, right = _make_operands(assignment[A], assignment[B], assignment[C], assignment[D], arr_name)
                op_name = op_assignments.get(op)
                if op_name is None:
                    continue
                result = _result_digits(op_name, left, right)
                if not all(ch in assignment for ch in out_str if ch != "-"):
                    continue
                expected = 0
                neg = out_str.startswith("-")
                for ch in (out_str[1:] if neg else out_str):
                    expected = expected * 10 + assignment[ch]
                if neg:
                    expected = -expected
                try:
                    if int(result) != expected:
                        return False
                except ValueError:
                    return False
            return True

        def backtrack(idx):
            iters[0] += 1
            if iters[0] % 5000 == 0 and _time.perf_counter() - start > timeout:
                return False
            if idx == len(syms_ordered):
                return _verify_all(ex_data, op_assignments, arr_name, assignment)
            sym = syms_ordered[idx]
            for d in range(10):
                if used[d]:
                    continue
                assignment[sym] = d
                used[d] = True
                if forward_check():
                    if backtrack(idx + 1):
                        return True
                del assignment[sym]
                used[d] = False
            return False

        if backtrack(0):
            return assignment.copy()
        return None


def _compute_answer(qA, qB, qOp, qC, qD, op_assignments, arr_name, mapping):
    a_d, b_d = mapping[qA], mapping[qB]
    c_d, d_d = mapping[qC], mapping[qD]
    left, right = _make_operands(a_d, b_d, c_d, d_d, arr_name)
    q_op = op_assignments.get(qOp, "add")
    result_str = _result_digits(q_op, left, right)
    rev_map = {v: k for k, v in mapping.items()}
    answer = _to_syms(result_str, rev_map)
    if answer is not None:
        return (q_op, arr_name, answer, mapping)
    return None


def _to_syms(digits, rev_map):
    result = []
    for ch in digits:
        if ch == "-":
            result.append("-")
            continue
        try:
            d = int(ch)
        except ValueError:
            return None
        if d not in rev_map:
            return None
        result.append(rev_map[d])
    return "".join(result)


def reasoning_cryptarithm(problem: Problem) -> Optional[str]:
    result = _try_solve(problem.examples, str(problem.question))
    if result is not None:
        op_name, arr_name, answer, mapping = result
        if mapping:
            return _format_cot(problem, op_name, arr_name, answer, mapping)
        else:
            return _format_concat_cot(problem, answer, op_name)
    return _reasoning_cryptarithm_concat(problem)


def _format_cot(problem, op_name, arr_name, answer, mapping):
    lines = ["We need to decode the symbol-digit cipher.",
             "I will put my final answer inside \\boxed{}.", ""]
    rev = {v: k for k, v in mapping.items()}
    lines.append("Mapping:")
    for d in range(10):
        if d in rev:
            lines.append(f"  {rev[d]} = {d}")
    lines.append("")
    for ex in problem.examples:
        inp = str(ex.input_value)
        A, B, op, C, D = inp[0], inp[1], inp[2], inp[3], inp[4]
        lines.append(f"  {inp} -> {ex.output_value}")
        lines.append(f"    = ({mapping[A]}{mapping[B]}) {op} ({mapping[C]}{mapping[D]})")
    lines.append(f"\nOperation: {op_name}\nQuestion: {problem.question} -> {answer}")
    lines.append("")
    lines.append("I will now return the answer in \\boxed{}")
    lines.append(f"The answer in \\boxed{{–}} is \\boxed{{{answer}}}")
    return "\n".join(lines)


def _format_concat_cot(problem, answer, op_name):
    direction = "concatenation" if op_name == "cat" else "reverse concatenation"
    lines = ["We need to infer the transformation rule.",
             "I will put my final answer inside \\boxed{}.", ""]
    for ex in problem.examples:
        inp = str(ex.input_value); out = str(ex.output_value)
        fwd = inp[0]+inp[1]+inp[3]+inp[4]; rev = inp[3]+inp[4]+inp[0]+inp[1]
        lines.append(f"  {inp} -> {out}")
        lines.append(f"    fwd={fwd} {'√' if out==fwd else '×'} rev={rev} {'√' if out==rev else '×'}")
    lines.append(f"\n{direction} → {answer}\n")
    lines.append("I will now return the answer in \\boxed{}")
    lines.append(f"The answer in \\boxed{{–}} is \\boxed{{{answer}}}")
    return "\n".join(lines)


def _reasoning_cryptarithm_concat(problem: Problem) -> Optional[str]:
    from dataclasses import dataclass
    @dataclass
    class _Ex:
        a: tuple[str, str]; op: str; b: tuple[str, str]; out: str
    def quote(s): return f"【{s}】"
    def _box(s): return "".join(f"【{c}】" for c in s)
    def _concat_type(exs):
        if all(ex.out == ex.a[0]+ex.a[1]+ex.b[0]+ex.b[1] for ex in exs): return "fwd"
        if all(ex.out == ex.b[0]+ex.b[1]+ex.a[0]+ex.a[1] for ex in exs): return "rev"
        return None
    exs = []
    for ex in problem.examples:
        inp = str(ex.input_value)
        if len(inp) != 5: return None
        exs.append(_Ex(a=(inp[0],inp[1]), op=inp[2], b=(inp[3],inp[4]), out=str(ex.output_value)))
    q = str(problem.question)
    if len(q) != 5: return None
    by_op = {}
    for e in exs: by_op.setdefault(e.op, []).append(e)
    concat_types = {op: _concat_type(ops) for op, ops in by_op.items() if _concat_type(ops)}
    if q[2] not in concat_types:
        if concat_types:
            most_common = max(concat_types, key=lambda op: len(by_op[op]))
            q_ct = concat_types[most_common]
        else: return None
    else:
        q_ct = concat_types[q[2]]
    answer = q[0]+q[1]+q[3]+q[4] if q_ct=="fwd" else q[3]+q[4]+q[0]+q[1]
    lines = ["We need to infer the transformation rule.",
             "I will put my final answer inside \\boxed{}.", ""]
    for ex, e in zip(problem.examples, exs):
        fwd = e.a[0]+e.a[1]+e.b[0]+e.b[1]; rev = e.b[0]+e.b[1]+e.a[0]+e.a[1]
        out = str(ex.output_value)
        lines.append(f"{quote(ex.input_value)} = {quote(ex.output_value)}")
        lines.append(f"  fwd={_box(fwd)} {'match' if out==fwd else 'mismatch'} rev={_box(rev)} {'match' if out==rev else 'mismatch'}")
        lines.append("")
    direction = "concatenation" if q_ct=="fwd" else "reverse concatenation"
    lines.append(f"Question: {quote(q)} → {answer}\n")
    lines.append("I will now return the answer in \\boxed{}")
    lines.append(f"The answer in \\boxed{{–}} is \\boxed{{{answer}}}")
    return "\n".join(lines)
