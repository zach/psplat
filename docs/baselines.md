<!-- psplat-docs: audience=dev -->
# Baseline importance scores

psplat's perceptual ordering (P2) has to beat what the pruning literature
already does. P1.1 ([#20](https://github.com/zach/psplat/issues/20)) implements
those methods' importance scores as orderings, in
`psplat/analyze/scores_baseline.py`. Each one is a pure function of the scene
and the arrays in `scene.weights.npz`, returning one score per splat, highest
first.

The papers leave choices open. Following handoff §8.9, a method whose reading
here is simpler than, or departs from, the paper carries a `-simple` suffix,
and this page records every choice that was made.

## Getting the inputs

Three of the scores need statistics beyond the contribution matrix. The
`psplat analyze` terms `hits`, `top` and `fisher` add them; `baselines`
asks for all of them at once:

```bash
psplat analyze tests/fixtures/synthetic/scene.ply --cameras tests/fixtures/synthetic/cams.json --backend torch --terms baselines --out fixture.weights.npz
```

```powershell
psplat analyze tests\fixtures\synthetic\scene.ply --cameras tests\fixtures\synthetic\cams.json --backend torch --terms baselines --out fixture.weights.npz
```

| Term | Array in `weights.npz` | Shape | Definition |
|---|---|---|---|
| `contribution` | `contribution` | `(N, V)` float32 | `Σ_pixels α·T` per view; always computed |
| `hits` | `hit_transmittance` | `(N,)` float32 | `Σ T` over every pixel, in every view, where the splat is composited |
| `top` | `top_count` | `(N,)` int64 | pixels, over all views, where the splat deposits the largest `α·T` |
| `fisher` | `fisher` | `(N, 6, 6)` float32 | `Σ_pixels g gᵀ`, `g = T · dα/dθ`, θ = world mean then log-scale |

"Composited" means the rasteriser actually blends it: alpha at least 1/255,
and the pixel has not stopped (see `docs/backends.md`). Only the torch
backend provides these statistics so far. Its results agree with a literal
per-pixel loop, and the Fisher with central finite differences, in
`tests/test_torch_backend.py`. On a CPU, `hits` and `top` add 10–40% to a
view's time, and `fisher` makes it 2–3 times as long.

## The baselines

| Name | Needs | Score |
|---|---|---|
| `opacity` | scene only | activated opacity σ_i |
| `contribution` | `contribution` | `Σ_v c_iv` |
| `lightgaussian` | `hits` | `σ_i · hit_transmittance_i · γ(V_i)` |
| `pup-simple` | `fisher` | `log det(I + F_i / ε)` |
| `coreset-simple` | `contribution` | sampling order `∝ s_i = max_v c_iv / Σ_j c_jv` |
| `minisplatting-simple` | `contribution`, `top` | top contributors first; each tier in sampling order `∝ Σ_v c_iv` |

### Sampling as an order

Coreset and Mini-Splatting choose which splats to keep by random sampling,
not by ranking. As an order, that becomes "the order in which sampling
without replacement, with probability proportional to the weight, would
draw the splats". Their scores are Gumbel keys, `log w_i + G_i`. Sorting
those keys gives exactly that order, so the first k splats are a weighted
sample of size k. That is also the set of distinct splats that sampling with
replacement turns up first. The draw is fixed by `seed` (default 0), and a
zero weight scores `-inf`, last.

### `opacity`

Activated opacity, `sigmoid(opacity)`. It ignores visibility and size, so an
off-screen splat keeps its opacity score. That is the point of including it:
it is the floor any rendering-aware score should beat.

### `contribution`

The `c_iv` that psplat's own perceptual weight is built on, summed over
views with equal weight. It is also Mini-Splatting's importance and the
coreset paper's query weight before normalisation.

### `lightgaussian`

LightGaussian (Fan et al., arXiv 2311.17245), global significance:

    GS_j = Σ_{i=1..MHW} 1(G(X_j), r_i) · σ_j · γ(Σ_j) · T

summed over all pixels `r_i` of all training views, with the volume factor
`γ(Σ) = min(V / V_max90, 1)^β`, `V = (4/3)π abc` from the three scales,
`V_max90` the 90th percentile of all splats' volumes, and `β = 0.1`.

| Question | Paper | Reading here |
|---|---|---|
| Is transmittance in the sum? | Eq. 1 includes `T`; the pseudocode (Algorithm 2) leaves it out | Eq. 1, with `T` |
| What counts as a ray "hitting" a splat? | `1(G(X_j), r_i)`, not further defined | the rasteriser composites it: alpha ≥ 1/255 and the pixel has not stopped |
| Percentile of what? | "90% largest of all sorted Gaussians" | of every splat in the scene, visible or not |
| Pruning ratio, distillation, fine-tuning | part of the method | not used; only the score's order is |

No ambiguity was simplified away, so the name has no suffix. It is still the
weakest-sourced choice here, since the paper's pseudocode and its equation
disagree, and the released code was not checked.

### `pup-simple`

PUP 3D-GS (Hanson et al., arXiv 2406.10219) scores each splat by the log
determinant of the block of the reconstruction loss's Fisher information
(its Hessian approximation) over that splat's mean and scale:
`U_i = log |∇²_{x_i, s_i} L|`, with `∇²L ≈ Σ_pixels ∇I ∇Iᵀ`.

| Question | Paper | Reading here |
|---|---|---|
| Scale parameterisation | not stated | log-scale, which is what 3DGS optimises |
| Which gradient of the image? | the full one | **direct term only:** `T · dα_i/dθ`. Transmittance in front is held fixed, and the term for how changing α_i dims the splats behind it is dropped |
| Colour | the RGB image | a single grey channel: the gradient of alpha, not of colour |
| Resolution | 4×4 patches at reduced resolution | every pixel at full resolution |
| A splat seen in fewer than six independent pixel gradients | singular, `log det = -∞` | damped: `log det(I + F/ε)`, `ε = 10⁻⁶ ×` the median of `trace(F)/6` over splats with any gradient |
| Clamped alpha (0.999) | not discussed | zero gradient there, as in the rasteriser's backward pass |

Damping and the division by ε shift every score by the same constant
`6 log ε`, so the order is that of `log det(F + εI)`. A splat with no
gradient, such as an invisible one, scores exactly 0.

### `coreset-simple`

Coreset sensitivity sampling (arXiv 2607.02721) defines a splat's
sensitivity as its largest share of any query, a query being one view, pixel
and colour channel:

    s(g_i) = max_q a(g_i, q) / A(q),   a = c_i · α_i · T_i,   A = Σ_j a

then samples splats i.i.d. with probability proportional to `s` and
reweights the opacity of what it keeps.

| Question | Paper | Reading here |
|---|---|---|
| Query granularity | (view, pixel, channel) | **one view**: `s_i = max_v c_iv / Σ_j c_jv` |
| Colour in the query weight | yes, with a no-colour variant (`α·T`) | the no-colour variant |
| Reweighting kept splats | part of the method | not used: psplat reorders and never edits attributes |

The per-pixel share is degenerate as an order: any splat that is the only one
covering some edge pixel scores 1. The per-view share still separates them.

### `minisplatting-simple`

Mini-Splatting (Fang and Wang, arXiv 2403.14166) simplifies a scene in two
ways: *intersection preserving* keeps, for each pixel of each view, the splat
with the largest blending weight; *sampling* draws splats with probability
proportional to importance `I_i = Σ_rays w_ij`, the summed blending weight.

| Question | Paper | Reading here |
|---|---|---|
| How the two combine into one order | separate steps, in a training schedule | two tiers: every splat that is the top contributor of at least one pixel comes first, then the rest; within each tier, the sampling order `∝ I` |
| Blending weight | `α·T` | `α·T`, so `I_i = Σ_v c_iv` |
| Ties for top contributor | not discussed | the nearer splat wins |
| Depth reinitialisation, densification | part of the method | not used |

## Properties the tests pin down

`tests/test_scores_baseline.py` checks each score against its definition on
hand-built arrays, and on scenes rendered by the torch backend.

**Invisible splats.** Every score except `opacity` ranks a splat that no view
sees last: 0 for the ranked scores, `-inf` for the sampled ones. On the
synthetic fixture, the five off-screen splats score exactly that.

**Splitting a splat.** Replace one splat with two coincident copies whose
opacities compose to the original's, `1 - (1 - o')² = o`, so the image barely
changes. A score that measures what a splat adds should give each copy about
half. Measured with `o = 0.3` over two views:

| Score | Each copy, as a fraction of the original |
|---|---|
| `opacity` | 0.54, 0.54 |
| `contribution` | 0.54, 0.49 |
| `lightgaussian` | 0.47, 0.45 |
| coreset sensitivity | about half (checked to 12%) |
| Fisher trace (`pup-simple`) | 0.30, 0.25 |

The split is exact only to first order in alpha, the copy behind sees
slightly less light, and each faint copy loses a little more of its tail to
the 1/255 threshold. Hence 0.54 and 0.49 rather than 0.5. PUP does not halve:
each copy's gradient is about half the original's, so its Fisher block is
about a quarter. A plain copy with the same opacity is a different case: it
doubles the alpha, and a faint splat's copy keeps nearly its whole
contribution.

## Not reproduced

All four papers fine-tune or retrain after pruning, and some reweight or
re-initialise what they keep. psplat's product is one ordering of an
unchanged scene that can be cut at any budget, so only each method's
selection criterion is reproduced here. Comparisons in E1 (P1.11) are of
orderings, not of the papers' full pipelines, and the report must say so.
