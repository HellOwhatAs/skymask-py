# skymask-py

<p align="center">
    <img src="https://github.com/user-attachments/assets/74c77624-0aca-444f-b1c0-8dad03d7821c" width="40%"/>
    <img src="https://github.com/user-attachments/assets/c3aa06ec-6e9b-4468-bd60-18f1b68af931" width="40%"/>
</p>
<p align="center">
    <a href="https://pypi.org/project/skymask-py/"><img src="https://img.shields.io/pypi/v/skymask-py"/></a>
    <a href="https://crates.io/crates/skymask-rs"><img src="https://img.shields.io/crates/v/skymask-rs" alt="crates.io"></a>
    <a href="https://docs.rs/skymask-rs/"><img src="https://docs.rs/skymask-rs/badge.svg" alt="docs"></a>
    <a href="https://github.com/HellOwhatAs/skymask-py/"><img src="https://img.shields.io/github/languages/top/HellOwhatAs/skymask-py"></a>
</p>

Compute piecewise analytical solutions of skymask for given polyhedra.  
Provides efficient algorithms, parallel computing, and sampling methods.  
> Python binding of rust crate [skymask-rs](https://github.com/HellOwhatAs/Skymask-rs/).

## Benchmark
Runs on 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz (8 Physical Cores / 16 Logical Threads) and NVIDIA GeForce RTX 3070 Laptop GPU.
The benchmark code is available at [benchmark.py](https://github.com/HellOwhatAs/Skymask/blob/main/benchmark.py).

|Method|Fps|Time Complexity|
|-|-|-|
|Parallel sampling: `World.par_samples`|1743.54|$O((k \cdot n \log n + m) \cdot \log r)$|
|Sequential sampling: `SkymaskMap.samples`|187.77|$O((k \cdot n \log n + m) \cdot \log r)$|
|[Naive approach](https://github.com/HellOwhatAs/Skymask/blob/main/skymask.py) with Cupy|84.98|$O(m \cdot n)$|
|[Naive approach](https://github.com/HellOwhatAs/Skymask/blob/main/skymask.py) with Numpy|4.91|$O(m \cdot n)$|

> Where $n$ represents the number of line segments, and $k$ denotes the average number of segments each line overlaps with in the analytical result.
> $r$ denotes the number of segments in the analytical result, and $m$ refers to the number of discrete sample points taken from the skymask.  

## Install
```
pip install skymask-py
```
Precompiled wheels **also** at https://github.com/HellOwhatAs/skymask-py/releases.

## Example
```py
import skymask_py
import numpy as np
import matplotlib.pyplot as plt

lines = np.array([
    #  xa,   ya,   za,   xb,   yb,   zb
    [ 1.0,  1.0,  1.0, -1.0,  1.0,  1.0],
    [-1.0,  1.0,  1.0, -1.0, -1.0,  1.0],
    [-1.0, -1.0,  1.0,  1.0, -1.0,  1.0],
    [ 1.0, -1.0,  1.0,  1.0,  1.0,  1.0],
])
world = skymask_py.World.from_lines(lines, np.inf)
for pos in [(0, 0), (0.5, 0)]:
    skymask = world.skymask(pos)
    print(f"\nskymask at {pos}")
    print("\n".join(
        f"pi/2-atan({a}*cos(t) + {b}*sin(t)) if t in [{s}, {e})"
        for (s, e), (a, b) in skymask.segments()
    ))

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    ax.set_title(f"skymask at {pos}")
    ax.yaxis.set_major_formatter("")
    x = np.linspace(-np.pi, np.pi, num=500, endpoint=True)
    x[-1] = x[0]
    y = np.pi / 2 - skymask.samples(x)
    ax.plot(x, y)
    ax.fill_between(x, y, np.pi / 2, alpha=0.2)
    ax.set_ylim(0, np.pi / 2)
plt.show()
```

<details open>
<summary>Outputs</summary>

```
skymask at (0, 0)
pi/2-atan(-1.0*cos(t) + -0.0*sin(t)) if t in [-3.141592653589793, -2.356194490192345)
pi/2-atan(-0.0*cos(t) + -1.0*sin(t)) if t in [-2.356194490192345, -0.7853981633974483)
pi/2-atan(1.0*cos(t) + -0.0*sin(t)) if t in [-0.7853981633974483, 0.7853981633974483)
pi/2-atan(-0.0*cos(t) + 1.0*sin(t)) if t in [0.7853981633974483, 2.356194490192345)
pi/2-atan(-1.0*cos(t) + -0.0*sin(t)) if t in [2.356194490192345, 3.141592653589793)

skymask at (0.5, 0)
pi/2-atan(-0.6666666666666666*cos(t) + -0.0*sin(t)) if t in [-3.141592653589793, -2.5535900500422257)
pi/2-atan(-0.0*cos(t) + -1.0*sin(t)) if t in [-2.5535900500422257, -1.1071487177940904)
pi/2-atan(2.0*cos(t) + -0.0*sin(t)) if t in [-1.1071487177940904, 1.1071487177940904)
pi/2-atan(-0.0*cos(t) + 1.0*sin(t)) if t in [1.1071487177940904, 2.5535900500422257)
pi/2-atan(-0.6666666666666666*cos(t) + -0.0*sin(t)) if t in [2.5535900500422257, 3.141592653589793)
```

<p align="center">
<img src="https://github.com/user-attachments/assets/4402d510-a529-4135-b7b2-0fec145dc0cb" width="45%"/> <img src="https://github.com/user-attachments/assets/8265abed-2fe6-4af9-91e3-0cc6e3b4957b" width="45%"/>
</p>
</details>
