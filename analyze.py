#!/usr/bin/env python3
"""Reads results.csv (+ correctness.csv, env.txt) and produces:
   summary.csv, fig_runtime_linear.png, fig_runtime_loglog.png, fig_memory.png, generated.tex
   (LaTeX tables + macros that report.tex \\input's, so re-running the benchmark and
   recompiling the report updates every number automatically)."""
import csv
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
IMPLS = ["Python", "Java-cold", "Java-warm", "C"]
LABEL = {"Python": "Python (CPython)", "Java-cold": "Java (cold JVM)",
         "Java-warm": "Java (3 warm-up runs)", "C": "C (gcc -O3)"}
# colorblind-safe (Okabe-Ito) + distinct markers/line styles so identity is not colour-only
STYLE = {"Python": ("#D55E00", "o", "-"), "Java-cold": ("#E69F00", "s", "--"),
         "Java-warm": ("#009E73", "D", "-"), "C": ("#0072B2", "^", "-")}
FIT_MIN_N = 200   # exponent fit uses n >= 200 (start-up/timer effects dominate below)


def load():
    data = {}
    status = {}
    with open(os.path.join(HERE, "results.csv")) as f:
        for r in csv.DictReader(f):
            key = (r["impl"], int(r["n"]))
            if r["status"] == "ok":
                d = data.setdefault(key, {"t": [], "rss": [], "ok": []})
                d["t"].append(float(r["seconds"]))
                d["rss"].append(int(r["maxrss_kb"]) / 1024)
                d["ok"].append(r["checksum_ok"] == "True")
            else:
                status[key] = r["status"]
    return data, status


def fmt(x):
    if x >= 100: return f"{x:.0f}"
    if x >= 1: return f"{x:.3f}"
    if x >= 0.001: return f"{x:.4f}"
    return f"{x:.5f}"


def main():
    data, status = load()
    sizes = sorted({n for (_, n) in list(data) + list(status)})
    stats = {}
    for k, d in data.items():
        t = np.array(d["t"])
        stats[k] = dict(reps=len(t), med=np.median(t), mn=t.min(), mx=t.max(),
                        sd=t.std(ddof=1) if len(t) > 1 else 0.0,
                        cv=100 * (t.std(ddof=1) / t.mean() if len(t) > 1 else 0.0),
                        rss=np.median(d["rss"]), allok=all(d["ok"]))

    with open(os.path.join(HERE, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["impl", "n", "reps", "median_s", "min_s", "max_s", "std_s", "cv_pct", "median_rss_MiB", "all_checksums_ok"])
        for impl in IMPLS:
            for n in sizes:
                s = stats.get((impl, n))
                if s: w.writerow([impl, n, s["reps"], s["med"], s["mn"], s["mx"], s["sd"], f"{s['cv']:.2f}", f"{s['rss']:.2f}", s["allok"]])

    # ---- exponents (log-log fit of median time, n >= FIT_MIN_N)
    exps = {}
    for impl in IMPLS:
        pts = [(n, stats[(impl, n)]["med"]) for n in sizes if (impl, n) in stats and n >= FIT_MIN_N]
        if len(pts) >= 3:
            x = np.log([p[0] for p in pts]); y = np.log([p[1] for p in pts])
            slope, icpt = np.polyfit(x, y, 1)
            r2 = 1 - ((y - (slope * x + icpt)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
            exps[impl] = (slope, r2, len(pts))

    # ---- figures
    def lineplot(logscale, fname):
        fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=200)
        for impl in IMPLS:
            ns = [n for n in sizes if (impl, n) in stats]
            if not ns: continue
            med = np.array([stats[(impl, n)]["med"] for n in ns])
            lo = med - np.array([stats[(impl, n)]["mn"] for n in ns])
            hi = np.array([stats[(impl, n)]["mx"] for n in ns]) - med
            c, m, ls = STYLE[impl]
            ax.errorbar(ns, med, yerr=[lo, hi], color=c, marker=m, ls=ls, lw=1.6, ms=5,
                        capsize=2, label=LABEL[impl])
        if logscale:
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xticks(sizes); ax.set_xticklabels(sizes, rotation=45, fontsize=8)
            ax.minorticks_off()
            ref_n = np.array([sizes[0], sizes[-1]], dtype=float)
            anchor = stats.get(("C", 200), stats.get(("Python", 200)))["med"]
            ax.plot(ref_n, anchor * (ref_n / 200) ** 3, color="gray", lw=1, ls=":", label=r"slope 3 ($\propto n^3$) reference")
            ax.set_title("Runtime vs matrix size (log-log; bars = min-max of repetitions)", fontsize=10)
        else:
            ax.set_title("Runtime vs matrix size (linear; bars = min-max of repetitions)", fontsize=10)
        ax.set_xlabel("Matrix size n (n x n)"); ax.set_ylabel("Multiplication time (s)")
        ax.grid(True, color="#dddddd", lw=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(HERE, fname)); plt.close(fig)

    lineplot(False, "fig_runtime_linear.png")
    lineplot(True, "fig_runtime_loglog.png")

    fig, ax = plt.subplots(figsize=(7.5, 4.0), dpi=200)
    for impl in IMPLS:
        ns = [n for n in sizes if (impl, n) in stats]
        if not ns: continue
        c, m, ls = STYLE[impl]
        ax.plot(ns, [stats[(impl, n)]["rss"] for n in ns], color=c, marker=m, ls=ls, lw=1.6, ms=5, label=LABEL[impl])
    ax.set_yscale("log"); ax.set_xlabel("Matrix size n (n x n)"); ax.set_ylabel("Peak RSS (MiB, median)")
    ax.set_title("Peak resident memory vs matrix size", fontsize=10)
    ax.grid(True, color="#dddddd", lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_memory.png")); plt.close(fig)

    # ---- LaTeX
    L = []
    cols = "r" + "r" * len(IMPLS)
    head = " & ".join(["$n$"] + [LABEL[i] for i in IMPLS]) + r" \\"

    def cell(impl, n, fn):
        s = stats.get((impl, n))
        if s: return fn(s)
        return r"\textit{timeout}" if status.get((impl, n)) == "timeout" else (r"\textit{error}" if (impl, n) in status else "--")

    def table(name, caption, fn, small=False):
        L.append(r"\begin{table}[H]\centering" + (r"\small\resizebox{\textwidth}{!}{" if small else ""))
        L.append(r"\begin{tabular}{" + cols + r"}\toprule")
        L.append(head + r"\midrule")
        for n in sizes:
            L.append(" & ".join([str(n)] + [cell(i, n, fn) for i in IMPLS]) + r" \\")
        L.append(r"\bottomrule\end{tabular}" + ("}" if small else "") + r"\caption{" + caption + r"}\label{tab:" + name + r"}\end{table}")

    table("median", "Median multiplication time in seconds (5 repetitions per cell).", lambda s: fmt(s["med"]))
    table("var", "Variability: min / max (s) and coefficient of variation (std/mean, \\%).",
          lambda s: f"{fmt(s['mn'])} / {fmt(s['mx'])} ({s['cv']:.1f}\\%)", small=True)
    table("mem", "Peak resident set size (MiB, median of repetitions).", lambda s: f"{s['rss']:.1f}")

    # speed-up table (Python / C etc.) at largest common size
    common = [n for n in sizes if all((i, n) in stats for i in IMPLS)]
    nmax = common[-1] if common else sizes[0]
    L.append(r"\begin{table}[H]\centering\begin{tabular}{lrr}\toprule")
    L.append(r"Implementation & Median at $n=" + str(nmax) + r"$ (s) & Slowdown vs.\ C \\\midrule")
    for i in IMPLS:
        s = stats.get((i, nmax))
        if s: L.append(f"{LABEL[i]} & {fmt(s['med'])} & {s['med'] / stats[('C', nmax)]['med']:.1f}$\\times$ \\\\")
    L.append(r"\bottomrule\end{tabular}\caption{Relative cost at the largest size where all four configurations completed.}\label{tab:ratio}\end{table}")

    # exponent table
    L.append(r"\begin{table}[H]\centering\begin{tabular}{lrrr}\toprule")
    L.append(r"Implementation & Fitted exponent $b$ & $R^2$ & Points ($n\ge" + str(FIT_MIN_N) + r"$) \\\midrule")
    for i in IMPLS:
        if i in exps: L.append(f"{LABEL[i]} & {exps[i][0]:.2f} & {exps[i][1]:.4f} & {exps[i][2]} \\\\")
    L.append(r"\bottomrule\end{tabular}\caption{Slope of $\log t$ vs.\ $\log n$ (least squares on medians); the cubic model predicts $b=3$.}\label{tab:exp}\end{table}")

    # doubling ratios
    pairs = [(a, 2 * a) for a in sizes if 2 * a in sizes]
    L.append(r"\begin{table}[H]\centering\small\resizebox{\textwidth}{!}{\begin{tabular}{l" + "r" * len(pairs) + r"}\toprule")
    L.append("Implementation & " + " & ".join(f"$t({b})/t({a})$" for a, b in pairs) + r" \\\midrule")
    for i in IMPLS:
        row = []
        for a, b in pairs:
            row.append(f"{stats[(i, b)]['med'] / stats[(i, a)]['med']:.1f}" if (i, a) in stats and (i, b) in stats else "--")
        L.append(f"{LABEL[i]} & " + " & ".join(row) + r" \\")
    L.append(r"\bottomrule\end{tabular}}\caption{Observed growth when $n$ doubles (ideal cubic model: 8.0).}\label{tab:doubling}\end{table}")

    # correctness table
    if os.path.exists(os.path.join(HERE, "correctness.csv")):
        rows = list(csv.DictReader(open(os.path.join(HERE, "correctness.csv"))))
        L.append(r"\begin{table}[H]\centering\begin{tabular}{lrr}\toprule Implementation & Checks passed & Total \\\midrule")
        for impl in ("C", "Java", "Python"):
            rr = [r for r in rows if r["impl"] == impl]
            L.append(f"{impl} & {sum(r['pass'] == 'True' for r in rr)} & {len(rr)} \\\\")
        L.append(r"\bottomrule\end{tabular}\caption{Correctness tests (exact integer equality with the NumPy reference).}\label{tab:correct}\end{table}")

    # failures / limits + checksum summary
    allok = all(s["allok"] for s in stats.values())
    nruns = sum(s["reps"] for s in stats.values())
    lim = {i: max([n for n in sizes if (i, n) in stats], default=None) for i in IMPLS}
    fails = [(i, n, status[(i, n)]) for (i, n) in sorted(status, key=lambda k: (k[0], k[1]))]
    macros = {
        "AllChecksumsOK": "all" if allok else "NOT all",
        "NRuns": str(nruns),
        "LimitC": str(lim["C"]), "LimitJavaCold": str(lim["Java-cold"]), "LimitJavaWarm": str(lim["Java-warm"]),
        "LimitPython": str(lim["Python"]),
        "FailList": "; ".join(f"{LABEL[i]} at $n={n}$ ({s})" for i, n, s in fails) or "none",
        "NMaxCommon": str(nmax),
    }
    for i, key in (("Python", "ExpPython"), ("Java-cold", "ExpJavaCold"), ("Java-warm", "ExpJavaWarm"), ("C", "ExpC")):
        if i in exps: macros[key] = f"{exps[i][0]:.2f}"
    # raw env block
    env = open(os.path.join(HERE, "env.txt")).read() if os.path.exists(os.path.join(HERE, "env.txt")) else ""
    def grab(pat, default="n/a"):
        m = re.search(pat, env, re.M); return m.group(1).strip() if m else default
    macros.update({
        "EnvCPU": grab(r"Model name:\s*(.+)"), "EnvCores": grab(r"CPU\(s\):\s*(\d+)"),
        "EnvRAM": f"{int(grab(r'MemTotal:\s*(\d+)', '0')) / 1048576:.1f} GiB",
        "EnvOS": grab(r"^platform: (.+)"), "EnvGcc": grab(r"^gcc: (.+)"),
        "EnvJava": grab(r"^java: (.+)").split("\n")[0], "EnvPython": grab(r"^python: (.+)"),
    })
    def esc(s): return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")
    macro_tex = "\n".join(f"\\newcommand{{\\{k}}}{{{v if k in ('FailList',) else esc(v)}}}" for k, v in macros.items())
    open(os.path.join(HERE, "generated.tex"), "w").write(macro_tex + "\n")
    blob = "\n".join(L) + "\n"
    for m in re.finditer(r"\\begin\{table\}.*?\\label\{tab:(\w+)\}\\end\{table\}", blob, re.S):
        open(os.path.join(HERE, f"tab_{m.group(1)}.tex"), "w").write(m.group(0) + "\n")
    print("exponents:", {k: round(v[0], 3) for k, v in exps.items()})
    print("limits:", lim, "failures:", fails, "all checksums ok:", allok, "runs:", nruns)


if __name__ == "__main__":
    main()
