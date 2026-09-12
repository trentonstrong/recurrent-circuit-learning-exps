"""Finite-dimensional sanity checks for the neutral-exploration proposal; no training."""
from itertools import combinations, product
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T = ((np.arange(16)[:, None] >> np.arange(3, -1, -1)) & 1).astype(float)
A = np.vstack((np.ones(16), T.T))
moves = []
for pair in combinations(range(4), 2):
    others = [r for r in range(4) if r not in pair]
    for fixed in product((0, 1), repeat=2):
        d = np.zeros(16)
        for bits in product((0, 1), repeat=2):
            table = [0] * 4
            for r, value in zip(others, fixed):
                table[r] = value
            for r, value in zip(pair, bits):
                table[r] = value
            gate = sum(value << (3-r) for r, value in enumerate(table))
            d[gate] = 1 if bits[0] == bits[1] else -1
        moves.append(d)
D = np.stack(moves, axis=1)
assert D.shape == (16, 24)
assert np.array_equal(A @ D, np.zeros((5, 24)))
assert np.linalg.matrix_rank(D) == 11

def response(p, q, h):
    C = T - q
    J = C.T * p
    return -(J @ J.T) @ h

def response_jacobian_on_fiber(p, q, h):
    C = T - q
    return -2 * C.T * (p * (C @ h))

rng = np.random.default_rng(20260912)
max_q_error = 0.
max_direction_derivative_error = 0.
for _ in range(100):
    logits = rng.normal(0, 1.5, 16)
    p = np.exp(logits - logits.max())
    p /= p.sum()
    q, h = p @ T, rng.normal(size=4)
    d = D[:, rng.integers(24)]
    lo, hi = -p[d > 0].min(), p[d < 0].min()
    delta = .05 * rng.uniform(lo, hi)
    moved = p + delta*d
    assert moved.min() > 0
    z = np.log(moved)
    rebuilt = np.exp(z-z.max())
    rebuilt /= rebuilt.sum()
    max_q_error = max(max_q_error, float(np.max(np.abs(rebuilt @ T-q))))
    v = response(p,q,h)
    unit = v/np.linalg.norm(v)
    jac = (np.eye(4)-np.outer(unit,unit)) @ response_jacobian_on_fiber(p,q,h) / np.linalg.norm(v)
    eps = min(-lo,hi)*1e-3
    plus, minus = response(p+eps*d,q,h), response(p-eps*d,q,h)
    fd = (plus/np.linalg.norm(plus)-minus/np.linalg.norm(minus))/(2*eps)
    error = np.linalg.norm(fd-jac@d)/max(np.linalg.norm(jac@d),1e-12)
    max_direction_derivative_error = max(max_direction_derivative_error,float(error))
assert max_q_error < 1e-13
assert max_direction_derivative_error < 1e-4

spins = 2*T-1
h = np.array([1.,0,0,0])
q = np.full(4,.5)
examples = []
for c in (.5,-.5):
    p = (1+c*spins[:,0]*spins[:,1])/16
    assert np.array_equal(p@T,q)
    M_expected = np.diag(np.full(4,(1+c*c)/64))
    M_expected[0,1] = M_expected[1,0] = 2*c/64
    v = response(p,q,h)
    assert np.array_equal(v,-M_expected@h)
    examples.append({"correlation":c,"p":p.tolist(),"M":M_expected.tolist(),"v":v.tolist()})
va,vb = [np.array(x["v"]) for x in examples]
angle = np.degrees(np.arccos(va@vb/(np.linalg.norm(va)*np.linalg.norm(vb))))
assert np.isclose(np.linalg.norm(va),np.linalg.norm(vb))
assert np.isclose(angle,77.31961650818019)

p = np.full(16,1/16)
v = response(p,q,h)
unit = v/np.linalg.norm(v)
W = (np.eye(4)-np.outer(unit,unit)) @ response_jacobian_on_fiber(p,q,h) @ D / np.linalg.norm(v)
scores = np.sum(W*W,axis=0)
assert np.count_nonzero(scores) == 12
weighted_expected = np.sum(scores*scores)/scores.sum()
uniform_expected = scores.mean()

fig,ax = plt.subplots(figsize=(6.7,5.2),layout="constrained")
colors = ("#2368ac","#d65b38")
for x,col in zip(examples,colors):
    vv=np.array(x["v"])*64
    ax.annotate("",xy=vv[:2],xytext=(0,0),arrowprops=dict(arrowstyle="-|>",lw=2.8,color=col,mutation_scale=17))
    ax.text(vv[0]-.05,vv[1]+(.14 if vv[1]>0 else -.19),
            rf"$c={x['correlation']:+.1f}$",color=col,ha="right",fontsize=13)
ax.annotate("",xy=(-1.75,0),xytext=(0,0),arrowprops=dict(arrowstyle="->",lw=1.6,color="#59616c",linestyle="dashed"))
ax.text(-1.78,.08,r"$-h$",ha="right",fontsize=12,color="#59616c")
a0=np.arctan2(va[1],va[0]); a1=np.arctan2(vb[1],vb[0])
theta=np.linspace(a1,a0+2*np.pi,150)
ax.plot(.55*np.cos(theta),.55*np.sin(theta),color="#737b86",lw=1)
ax.text(-.93,.04,f"{angle:.1f}°",ha="center",fontsize=12)
ax.axhline(0,color="#d3d7dc",lw=.8,zorder=0)
ax.axvline(0,color="#d3d7dc",lw=.8,zorder=0)
ax.set(xlim=(-2,.25),ylim=(-1.45,1.45),aspect="equal",
       xlabel=r"Response in $q_{00}$  (units of $1/64$)",
       ylabel=r"Response in $q_{01}$  (units of $1/64$)")
fig.suptitle("Same gate function and response norm",fontsize=13)
ax.set_title(r"$q=(1/2,1/2,1/2,1/2)$ and $h=(1,0,0,0)$",fontsize=10,pad=12)
ax.spines[["top","right"]].set_visible(False)
out=Path(__file__).resolve().parent
fig.savefig(out/"response_directions.svg",metadata={"Date":None})
fig.savefig(out/"response_directions.png",dpi=160)
summary={
    "kind":"synthetic_algebra_sanity_check_not_training",
    "seed":20260912,
    "neutral_move_count":24,"neutral_span_rank":11,
    "max_reencoded_q_error":max_q_error,
    "max_direction_derivative_relative_error":max_direction_derivative_error,
    "equal_norm_example_angle_degrees":float(angle),
    "uniform_example_first_order_active_moves":int(np.count_nonzero(scores)),
    "uniform_example_score_weighted_vs_uniform_predicted_squared_angle_ratio":float(weighted_expected/uniform_expected),
    "examples":examples,
}
(out/"sanity_results.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps({k:v for k,v in summary.items() if k!="examples"},indent=2))
