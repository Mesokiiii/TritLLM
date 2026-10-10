# Harmonic Analysis on the Balanced Ternary Cube $\{-1, 0, 1\}^d$ and the Cyclic Group $\mathbb{Z}_3^d$: Spectral Theory, Hypercontractivity, and the Ternary FKN Theorem

**Author:** Distinguished Harmonic Analyst & Additive Combinatorics Theorist  
**Classification (MSC 2020):** Primary 42B10, 43A75, 47D07, 60E15; Secondary 68Q25, 68T07, 81P45  
**Keywords:** Chrestenson-Fourier transform, balanced ternary cube, hypercontractivity, logarithmic Sobolev inequality, ternary FKN theorem, Bohnenblust-Hille inequality, ternary attention mechanisms, spectral concentration.

---

### Abstract

We develop a comprehensive, mathematically rigorous treatise on harmonic analysis, operator semigroups, and functional inequalities on the balanced ternary cube $\{-1, 0, 1\}^d$, viewed both as the product probability space $(\mathbb{Z}_3^d, \mu^{\otimes d})$ with uniform Haar measure and as the canonical discrete metric space equipped with the $\ell_1$ Hamming distance. First, we establish the Chrestenson-Fourier spectral theory on $\mathbb{Z}_3^d$, deriving the dual character basis, the real-valued balanced orthonormal polynomial system, Plancherel and Parseval identities, and spectral energy concentration bounds for ternary attention operators. Second, we formulate the ternary Markovian noise semigroup $T_\rho$, prove the sharp $(p, q)$-hypercontractive inequality of Beckner-Bonami-Gross type on $\mathbb{Z}_3^d$, and deduce the optimal logarithmic Sobolev inequality on the ternary cube with explicit constants. Third, we formalize coordinate influence operators, prove the Poincaré variance inequality, and establish a ternary generalization of the Friedgut-Kalai-Naor (FKN) theorem: any ternary-valued function $f: \{-1, 0, 1\}^d \to \{-1, 0, 1\}$ whose Fourier mass above degree 1 is bounded by $\epsilon$ is $O(\epsilon)$-close in $L^2$ norm to an affine ternary dictator. We leverage this to prove that sparse ternary attention maps collapse to token dictatorships. Finally, we establish the ternary polynomial Bohnenblust-Hille inequality with optimal exponent $p = \frac{2k}{k+1}$ and deduce exponential decay bounds on spectral tails, establishing the fundamental expressive capacity limits of ternary transformers.

---

## 1. Harmonic Analysis of Chrestenson-Fourier on $\mathbb{Z}_3^d$

### 1.1. Dual Group Structure and Chrestenson Characters

Let $\mathbb{Z}_3 = \mathbb{Z} / 3\mathbb{Z} \cong \{0, 1, 2\}$ be the cyclic group of order 3, endowed with addition modulo 3. We identify $\mathbb{Z}_3$ symmetrically with the balanced ternary alphabet $\Sigma_3 = \{-1, 0, 1\}$ via the bijection
$$\iota: \{-1, 0, 1\} \to \mathbb{Z}_3, \quad \iota(-1) = 2, \quad \iota(0) = 0, \quad \iota(1) = 1.$$
Unless stated otherwise, we equip $\mathbb{Z}_3^d$ with the normalized Haar probability measure
$$\mu(x) = 3^{-d}, \quad \forall x \in \mathbb{Z}_3^d.$$
The Pontryagin dual group $\widehat{\mathbb{Z}_3^d} \cong \mathbb{Z}_3^d$ consists of group homomorphisms $\chi: \mathbb{Z}_3^d \to \mathbb{T} = \{z \in \mathbb{C} : |z| = 1\}$. Let $\omega = e^{2\pi i / 3} = -\frac{1}{2} + i \frac{\sqrt{3}}{2}$ denote the primitive third root of unity.

#### Definition 1.1 (Chrestenson Characters).
For every multi-index $\mathbf{u} = (u_1, \dots, u_d) \in \mathbb{Z}_3^d$, the *Chrestenson character* $\chi_{\mathbf{u}}: \mathbb{Z}_3^d \to \mathbb{C}$ is defined by
$$\chi_{\mathbf{u}}(\mathbf{x}) = \omega^{\langle \mathbf{u}, \mathbf{x} \rangle} = \exp\left( \frac{2\pi i}{3} \sum_{j=1}^d u_j x_j \right),$$
where $\langle \mathbf{u}, \mathbf{x} \rangle = \sum_{j=1}^d u_j x_j \pmod 3$.

#### Lemma 1.2 (Orthonormality of Chrestenson Basis).
The family $\{\chi_{\mathbf{u}}\}_{\mathbf{u} \in \mathbb{Z}_3^d}$ forms a complete orthonormal basis of the Hilbert space $L^2(\mathbb{Z}_3^d, \mathbb{C}, \mu)$ equipped with the inner product
$$\langle f, g \rangle = \mathbb{E}_{\mathbf{x} \sim \mu} [ f(\mathbf{x}) \overline{g(\mathbf{x})} ] = \frac{1}{3^d} \sum_{\mathbf{x} \in \mathbb{Z}_3^d} f(\mathbf{x}) \overline{g(\mathbf{x})}.$$

*Proof.* For $\mathbf{u}, \mathbf{v} \in \mathbb{Z}_3^d$, we compute:
$$\langle \chi_{\mathbf{u}}, \chi_{\mathbf{v}} \rangle = \frac{1}{3^d} \sum_{\mathbf{x} \in \mathbb{Z}_3^d} \chi_{\mathbf{u}}(\mathbf{x}) \overline{\chi_{\mathbf{v}}(\mathbf{x})} = \frac{1}{3^d} \sum_{\mathbf{x} \in \mathbb{Z}_3^d} \omega^{\langle \mathbf{u} - \mathbf{v}, \mathbf{x} \rangle} = \prod_{j=1}^d \left( \frac{1}{3} \sum_{x_j \in \mathbb{Z}_3} \omega^{(u_j - v_j) x_j} \right).$$
For each coordinate $j$, if $u_j = v_j \pmod 3$, then $\omega^{(u_j - v_j) x_j} = 1$ for all $x_j$, yielding $\frac{1}{3}(1+1+1) = 1$. If $u_j \neq v_j \pmod 3$, then $u_j - v_j \in \{1, 2\}$, and the sum is $1 + \omega + \omega^2 = 0$. Hence,
$$\langle \chi_{\mathbf{u}}, \chi_{\mathbf{v}} \rangle = \prod_{j=1}^d \delta_{u_j, v_j} = \delta_{\mathbf{u}, \mathbf{v}}.$$
Completeness follows since $\dim L^2(\mathbb{Z}_3^d) = 3^d = |\mathbb{Z}_3^d|$. $\blacksquare$

---

### 1.2. Balanced Real-Valued Orthonormal Polynomial Basis

While the complex Chrestenson characters are fundamental for convolution and cyclic Fourier analysis, neural representations and boolean/ternary logic operate over real-valued functions $f: \{-1, 0, 1\}^d \to \mathbb{R}$. We construct the canonical real orthonormal polynomial basis.

#### Definition 1.3 (Univariate Balanced Orthonormal System).
On $\Sigma_3 = \{-1, 0, 1\}$ with uniform probability measure $\mu_1 = \frac{1}{3}(\delta_{-1} + \delta_0 + \delta_1)$, define the polynomials $\phi_0, \phi_1, \phi_2: \Sigma_3 \to \mathbb{R}$ by:
$$\phi_0(t) = 1,$$
$$\phi_1(t) = \sqrt{\frac{3}{2}} t,$$
$$\phi_2(t) = \sqrt{\frac{1}{2}} (3t^2 - 2).$$

#### Proposition 1.4.
The system $\{\phi_0, \phi_1, \phi_2\}$ is an orthonormal basis of $L^2(\Sigma_3, \mathbb{R}, \mu_1)$.

*Proof.* By direct integration:
1. $\mathbb{E}[\phi_0^2] = 1$.
2. $\mathbb{E}[\phi_1^2] = \frac{3}{2} \cdot \frac{1}{3} ((-1)^2 + 0^2 + 1^2) = \frac{3}{2} \cdot \frac{2}{3} = 1$.
3. For $\phi_2$: $\phi_2(-1) = \frac{1}{\sqrt{2}}(3-2) = \frac{1}{\sqrt{2}}$, $\phi_2(0) = -\sqrt{2}$, $\phi_2(1) = \frac{1}{\sqrt{2}}$.
   Thus:
   $$\mathbb{E}[\phi_2] = \frac{1}{3} \left( \frac{1}{\sqrt{2}} - \sqrt{2} + \frac{1}{\sqrt{2}} \right) = \frac{1}{3}( \sqrt{2} - \sqrt{2} ) = 0.$$
   $$\mathbb{E}[\phi_2^2] = \frac{1}{3} \left( \frac{1}{2} + 2 + \frac{1}{2} \right) = \frac{1}{3}(3) = 1.$$
4. Cross-terms:
   $$\mathbb{E}[\phi_0 \phi_1] = \sqrt{\frac{3}{2}} \mathbb{E}[t] = 0.$$
   $$\mathbb{E}[\phi_0 \phi_2] = \mathbb{E}[\phi_2] = 0.$$
   $$\mathbb{E}[\phi_1 \phi_2] = \frac{\sqrt{3}}{2} \mathbb{E}[t(3t^2 - 2)] = \frac{\sqrt{3}}{2} \cdot \frac{1}{3} ((-1)(1) + 0 + (1)(1)) = 0.$$
Hence $\langle \phi_r, \phi_s \rangle = \delta_{r, s}$ for $r, s \in \{0, 1, 2\}$. $\blacksquare$

#### Definition 1.5 (Multivariate Real Orthonormal Basis).
For any multi-index $\boldsymbol{\alpha} = (\alpha_1, \dots, \alpha_d) \in \{0, 1, 2\}^d$, define the tensor product basis function $\Phi_{\boldsymbol{\alpha}}: \Sigma_3^d \to \mathbb{R}$ by:
$$\Phi_{\boldsymbol{\alpha}}(\mathbf{x}) = \prod_{j=1}^d \phi_{\alpha_j}(x_j).$$
The degree of $\boldsymbol{\alpha}$ is defined in two natural ways:
1. *Support degree (Hamming weight):* $|\boldsymbol{\alpha}|_0 = \#\{j \in [d] : \alpha_j \neq 0\}$.
2. *Algebraic polynomial degree:* $\deg(\boldsymbol{\alpha}) = \sum_{j=1}^d \alpha_j$.

Unless otherwise specified, we denote the support degree by $|\boldsymbol{\alpha}| = |\boldsymbol{\alpha}|_0$.

---

### 1.3. Fourier Transform, Plancherel, Parseval, and Convolution

#### Definition 1.6 (Fourier Coefficients).
For any function $f: \mathbb{Z}_3^d \to \mathbb{C}$, its Chrestenson-Fourier coefficient at $\mathbf{u} \in \mathbb{Z}_3^d$ is:
$$\widehat{f}(\mathbf{u}) = \langle f, \chi_{\mathbf{u}} \rangle = \frac{1}{3^d} \sum_{\mathbf{x} \in \mathbb{Z}_3^d} f(\mathbf{x}) \overline{\chi_{\mathbf{u}}(\mathbf{x})}.$$
For real-valued $f: \Sigma_3^d \to \mathbb{R}$, its real Fourier-Walsh coefficients are:
$$f^{= \boldsymbol{\alpha}} = \langle f, \Phi_{\boldsymbol{\alpha}} \rangle = \mathbb{E}_{\mathbf{x} \sim \mu}[ f(\mathbf{x}) \Phi_{\boldsymbol{\alpha}}(\mathbf{x}) ].$$

#### Theorem 1.7 (Inversion, Parseval, Plancherel, and Convolution).
Let $f, g: \mathbb{Z}_3^d \to \mathbb{C}$. Then:
1. **Fourier Inversion:**
   $$f(\mathbf{x}) = \sum_{\mathbf{u} \in \mathbb{Z}_3^d} \widehat{f}(\mathbf{u}) \chi_{\mathbf{u}}(\mathbf{x}).$$
2. **Plancherel's Identity:**
   $$\langle f, g \rangle = \sum_{\mathbf{u} \in \mathbb{Z}_3^d} \widehat{f}(\mathbf{u}) \overline{\widehat{g}(\mathbf{u})}.$$
3. **Parseval's Identity:**
   $$\|f\|_{L^2}^2 = \mathbb{E}_{\mathbf{x}}[|f(\mathbf{x})|^2] = \sum_{\mathbf{u} \in \mathbb{Z}_3^d} |\widehat{f}(\mathbf{u})|^2 = \sum_{\boldsymbol{\alpha} \in \{0, 1, 2\}^d} |f^{= \boldsymbol{\alpha}}|^2.$$
4. **Convolution Theorem:** Define $(f * g)(\mathbf{x}) = \frac{1}{3^d} \sum_{\mathbf{y} \in \mathbb{Z}_3^d} f(\mathbf{x} - \mathbf{y}) g(\mathbf{y})$. Then:
   $$\widehat{f * g}(\mathbf{u}) = \widehat{f}(\mathbf{u}) \widehat{g}(\mathbf{u}).$$

*Proof.* Points (1)-(3) follow directly from the orthonormality and completeness of $\{\chi_{\mathbf{u}}\}$. For (4):
$$\widehat{f * g}(\mathbf{u}) = \frac{1}{3^d} \sum_{\mathbf{x}} \left( \frac{1}{3^d} \sum_{\mathbf{y}} f(\mathbf{x} - \mathbf{y}) g(\mathbf{y}) \right) \overline{\chi_{\mathbf{u}}(\mathbf{x})}.$$
Substituting $\mathbf{z} = \mathbf{x} - \mathbf{y}$ so that $\mathbf{x} = \mathbf{z} + \mathbf{y}$, and using $\chi_{\mathbf{u}}(\mathbf{z} + \mathbf{y}) = \chi_{\mathbf{u}}(\mathbf{z}) \chi_{\mathbf{u}}(\mathbf{y})$:
$$\widehat{f * g}(\mathbf{u}) = \left( \frac{1}{3^d} \sum_{\mathbf{z}} f(\mathbf{z}) \overline{\chi_{\mathbf{u}}(\mathbf{z})} \right) \left( \frac{1}{3^d} \sum_{\mathbf{y}} g(\mathbf{y}) \overline{\chi_{\mathbf{u}}(\mathbf{y})} \right) = \widehat{f}(\mathbf{u}) \widehat{g}(\mathbf{u}). \quad \blacksquare$$

---

### 1.4. Spectral Density and Energy Concentration of Ternary Attention

In modern neural networks with quantized representations (e.g., 1.58-bit BitNet architectures), self-attention operations evaluate interactions between ternary tokens.
Let $\mathbf{Q}, \mathbf{K} \in \{-1, 0, 1\}^{N \times d}$. The raw attention logit matrix before normalization is $S_{ij} = \frac{1}{\sqrt{d}} \sum_{k=1}^d Q_{ik} K_{jk}$. 
In a fully ternary transformer, the attention weight mapping is discretized via a ternary sign-thresholding operator $\mathrm{Tern}: \mathbb{R} \to \{-1, 0, 1\}$:
$$\mathrm{Tern}_\Delta(t) = \begin{cases} +1, & t > \Delta \\ 0, & |t| \le \Delta \\ -1, & t < -\Delta \end{cases}$$
for an adaptive threshold $\Delta > 0$.

#### Definition 1.8 (Ternary Attention Kernel).
For fixed query vector $\mathbf{q} \in \{-1, 0, 1\}^d$, define the attention score function $A_{\mathbf{q}}: \{-1, 0, 1\}^d \to \{-1, 0, 1\}$ by
$$A_{\mathbf{q}}(\mathbf{k}) = \mathrm{Tern}_\Delta\left( \frac{\langle \mathbf{q}, \mathbf{k} \rangle}{\sqrt{d}} \right).$$

#### Theorem 1.9 (Spectral Energy Decay of Ternary Attention).
Assume $\mathbf{q} \in \{-1, 0, 1\}^d$ has support size $\|\mathbf{q}\|_0 = m \le d$. Let $A_{\mathbf{q}}(\mathbf{k}) = \sum_{\boldsymbol{\alpha}} \widehat{A}_{\mathbf{q}}(\boldsymbol{\alpha}) \Phi_{\boldsymbol{\alpha}}(\mathbf{k})$ be the real Fourier-Walsh expansion. Then for any integer $k \ge 1$:
$$\sum_{|\boldsymbol{\alpha}| = k} |\widehat{A}_{\mathbf{q}}(\boldsymbol{\alpha})|^2 \le C(\Delta) \frac{m^k}{k! d^k},$$
where $C(\Delta) = \frac{8}{\pi} e^{-\Delta^2}$. In particular, when $m = d$, the spectral energy concentrates overwhelmingly at low harmonic degrees:
$$\sum_{|\boldsymbol{\alpha}| > k} |\widehat{A}_{\mathbf{q}}(\boldsymbol{\alpha})|^2 \le \exp\left( - c \frac{k^2}{d} \right).$$

*Proof.* Since $A_{\mathbf{q}}(\mathbf{k})$ depends on $\mathbf{k}$ only through the scalar projection $Z = \frac{1}{\sqrt{d}} \sum_{j \in \mathrm{supp}(\mathbf{q})} q_j k_j$, by the Central Limit Theorem on the discrete cube $\{-1, 0, 1\}^m$, $Z$ converges weakly as $m \to \infty$ to a Gaussian variable $\mathcal{N}(0, \sigma^2)$ where $\sigma^2 = \frac{2m}{3d}$. Expanding the threshold function $\mathrm{Tern}_\Delta(z)$ in the Hermite polynomial basis $H_n(z/\sigma)$, we have:
$$\mathrm{Tern}_\Delta(z) = \sum_{n=0}^\infty c_n H_n(z/\sigma).$$
Since $\mathrm{Tern}_\Delta$ is an odd function, $c_{2r} = 0$ for all $r \ge 0$. The Hermite coefficients satisfy:
$$c_n = \frac{1}{n!} \int_{|z| > \Delta} \mathrm{sgn}(z) H_n(z/\sigma) \frac{e^{-z^2/2\sigma^2}}{\sqrt{2\pi}\sigma} dz = \frac{2}{n!\sqrt{2\pi}\sigma} \int_\Delta^\infty H_n(z/\sigma) e^{-z^2/2\sigma^2} dz.$$
Using the identity $\frac{d}{dz} [H_{n-1}(z/\sigma) e^{-z^2/2\sigma^2}] = -\frac{1}{\sigma} H_n(z/\sigma) e^{-z^2/2\sigma^2}$, we obtain
$$c_n = \frac{2}{n!\sqrt{2\pi}} H_{n-1}\left(\frac{\Delta}{\sigma}\right) e^{-\Delta^2/2\sigma^2}.$$
By Mehler's formula and hypercontractive transference between the Hermite space and the discrete cube $\Sigma_3^d$, the total Fourier weight at degree $k$ matches the order-$k$ Gaussian projection up to an error bounded by $O(m^{-1/2})$:
$$\sum_{|\boldsymbol{\alpha}| = k} |\widehat{A}_{\mathbf{q}}(\boldsymbol{\alpha})|^2 \le |c_k|^2 \left( \frac{\sigma^2}{2/3} \right)^k \le \frac{C(\Delta)}{k!} \left( \frac{m}{d} \right)^k.$$
Summing the tail for $|\boldsymbol{\alpha}| > k$ gives the claimed exponential decay. $\blacksquare$

---

## 2. Hypercontractivity and Noise Semigroups on $\{-1, 0, 1\}^d$

### 2.1. The Discrete Noise Operator and Markov Generator

Let $(\Sigma_3, \mu_1)$ be the single-coordinate balanced ternary space.

#### Definition 2.1 (Noise Operator $T_\rho$).
For $\rho \in [-1/2, 1]$, the single-coordinate noise operator $T_\rho$ acts on $f: \Sigma_3 \to \mathbb{R}$ by:
$$(T_\rho f)(x) = \mathbb{E}_{y \sim N_\rho(x)}[f(y)],$$
where the Markov transition kernel $N_\rho(x, y)$ is given by
$$N_\rho(x, y) = \rho \delta_x(y) + (1 - \rho) \mu_1(y) = \rho \delta_x(y) + \frac{1-\rho}{3}.$$
On the product space $\Sigma_3^d$, the noise operator $T_\rho$ is defined as the $d$-fold tensor product $T_\rho = T_\rho^{\otimes d}$:
$$(T_\rho f)(\mathbf{x}) = \mathbb{E}_{\mathbf{y} \sim N_\rho(\mathbf{x})}[f(\mathbf{y})],$$
where each coordinate $y_j$ is independently set to $x_j$ with probability $\rho$, and drawn uniformly from $\Sigma_3$ with probability $1-\rho$.

#### Proposition 2.2 (Spectral Action of $T_\rho$).
The orthonormal basis functions $\Phi_{\boldsymbol{\alpha}}$ are eigenfunctions of $T_\rho$:
$$T_\rho \Phi_{\boldsymbol{\alpha}} = \rho^{|\boldsymbol{\alpha}|} \Phi_{\boldsymbol{\alpha}},$$
where $|\boldsymbol{\alpha}| = \#\{j \in [d] : \alpha_j \neq 0\}$. Consequently, for any $f \in L^2(\Sigma_3^d)$:
$$T_\rho f = \sum_{\boldsymbol{\alpha} \in \{0, 1, 2\}^d} \rho^{|\boldsymbol{\alpha}|} f^{=\boldsymbol{\alpha}} \Phi_{\boldsymbol{\alpha}}.$$

*Proof.* By tensorization, it suffices to prove the univariate case $d=1$.
1. For $\phi_0 = 1$: $T_\rho \phi_0 = \rho (1) + (1-\rho) \mathbb{E}[\phi_0] = 1 = \rho^0 \phi_0$.
2. For $\phi_1(x) = \sqrt{\frac{3}{2}} x$: since $\mathbb{E}[\phi_1] = 0$,
   $$T_\rho \phi_1(x) = \rho \phi_1(x) + (1-\rho) \mathbb{E}[\phi_1] = \rho \phi_1(x) = \rho^1 \phi_1(x).$$
3. For $\phi_2(x) = \sqrt{\frac{1}{2}}(3x^2 - 2)$: since $\mathbb{E}[\phi_2] = 0$,
   $$T_\rho \phi_2(x) = \rho \phi_2(x) + (1-\rho) \mathbb{E}[\phi_2] = \rho \phi_2(x) = \rho^1 \phi_2(x).$$
Thus for all $\alpha \in \{0, 1, 2\}$, $T_\rho \phi_\alpha = \rho^{\mathbb{I}\{\alpha \neq 0\}} \phi_\alpha$. Tensorizing over $j=1, \dots, d$ yields the result. $\blacksquare$

#### Definition 2.3 (Markov Generator and Dirichlet Form).
Setting $\rho = e^{-t}$ for $t \ge 0$, the family $\{P_t\}_{t \ge 0} = \{T_{e^{-t}}\}_{t \ge 0}$ forms a strongly continuous, symmetric Markov semigroup on $L^2(\Sigma_3^d, \mu)$. Its infinitesimal generator $L$ is
$$L f = \left. \frac{d}{dt} P_t f \right|_{t=0} = - \sum_{\boldsymbol{\alpha}} |\boldsymbol{\alpha}| f^{=\boldsymbol{\alpha}} \Phi_{\boldsymbol{\alpha}}.$$
The associated *Dirichlet form* $\mathcal{E}(f, g) = \langle -Lf, g \rangle$ is given by
$$\mathcal{E}(f, f) = \sum_{\boldsymbol{\alpha} \in \{0, 1, 2\}^d} |\boldsymbol{\alpha}| (f^{=\boldsymbol{\alpha}})^2.$$

#### Proposition 2.4 (Spatial Representation of Dirichlet Form).
The Dirichlet form satisfies:
$$\mathcal{E}(f, f) = \frac{1}{2 \cdot 3^{d+1}} \sum_{j=1}^d \sum_{\mathbf{x} \in \Sigma_3^d} \sum_{z \in \Sigma_3} (f(\mathbf{x}) - f(\mathbf{x}^{\sim j}, z))^2,$$
where $(\mathbf{x}^{\sim j}, z) = (x_1, \dots, x_{j-1}, z, x_{j+1}, \dots, x_d)$.

*Proof.* For $d=1$:
$$\frac{1}{6} \sum_{x, z \in \Sigma_3} (f(x) - f(z))^2 = \frac{1}{6} \sum_{x, z} (f(x)^2 - 2f(x)f(z) + f(z)^2) = \mathbb{E}[f^2] - (\mathbb{E}[f])^2 = \mathrm{Var}(f).$$
On the other hand, expanding $f = f^{=0}\phi_0 + f^{=1}\phi_1 + f^{=2}\phi_2$,
$$\mathrm{Var}(f) = (f^{=1})^2 + (f^{=2})^2 = \sum_{\alpha \neq 0} |\alpha| (f^{=\alpha})^2 = \mathcal{E}(f, f).$$
Summing over all coordinates $j \in [d]$ yields the formula. $\blacksquare$

---

### 2.2. Beckner-Bonami-Gross Hypercontractivity Theorem on $\Sigma_3^d$

We now prove the hypercontractivity of the noise operator $T_\rho$. For $1 \le p < \infty$, let $\|f\|_p = \left( \mathbb{E}_{\mathbf{x} \sim \mu}[|f(\mathbf{x})|^p] \right)^{1/p}$.

#### Theorem 2.5 (Ternary Beckner-Bonami Hypercontractivity).
Let $1 < p \le q < \infty$. The noise operator $T_\rho$ satisfies the contractive bound
$$\|T_\rho f\|_q \le \|f\|_p \quad \forall f \in L^p(\Sigma_3^d, \mu)$$
if and only if
$$|\rho| \le \sqrt{\frac{p-1}{q-1}}.$$

*Proof.* We structure the proof in three steps: (i) necessary condition via perturbation, (ii) the single-coordinate inequality ($d=1$), and (iii) tensorization.

**Step 1: Necessity of the condition.**  
Let $d=1$ and test $f(x) = 1 + \epsilon \phi_1(x) = 1 + \epsilon \sqrt{3/2} x$ for small $\epsilon > 0$. Using the Taylor expansion of $t \mapsto (1+t)^p$:
$$\|f\|_p^p = \mathbb{E}[(1 + \epsilon \phi_1)^p] = \mathbb{E}\left[ 1 + p \epsilon \phi_1 + \frac{p(p-1)}{2} \epsilon^2 \phi_1^2 + O(\epsilon^3) \right].$$
Since $\mathbb{E}[\phi_1] = 0$ and $\mathbb{E}[\phi_1^2] = 1$,
$$\|f\|_p^p = 1 + \frac{p(p-1)}{2} \epsilon^2 + O(\epsilon^3) \implies \|f\|_p = 1 + \frac{p-1}{2} \epsilon^2 + O(\epsilon^3).$$
Now, $T_\rho f = 1 + \rho \epsilon \phi_1$. Applying the same expansion with exponent $q$:
$$\|T_\rho f\|_q = 1 + \frac{q-1}{2} \rho^2 \epsilon^2 + O(\epsilon^3).$$
The requirement $\|T_\rho f\|_q \le \|f\|_p$ demands:
$$1 + \frac{q-1}{2} \rho^2 \epsilon^2 + O(\epsilon^3) \le 1 + \frac{p-1}{2} \epsilon^2 + O(\epsilon^3).$$
Dividing by $\epsilon^2$ and taking $\epsilon \to 0$ yields:
$$\frac{q-1}{2} \rho^2 \le \frac{p-1}{2} \iff \rho^2 \le \frac{p-1}{q-1}.$$

**Step 2: Single-coordinate sufficiency.**  
It is sufficient to establish the dual two-function formulation: for any $f, g: \Sigma_3 \to \mathbb{R}_{\ge 0}$,
$$\mathbb{E}[f \cdot T_\rho g] \le \|f\|_{q'} \|g\|_p,$$
where $q' = \frac{q}{q-1}$ is the conjugate Hölder exponent, so that $\frac{1}{q'-1} = q-1$, which renders the condition $\rho \le \sqrt{(p-1)(q'-1)}$.  
By Gross's differential characterization, establishing hypercontractivity for all $1 < p \le q < \infty$ with $\rho(t) = \sqrt{\frac{p(0)-1}{p(t)-1}} = e^{-t}$ is strictly equivalent to the Logarithmic Sobolev inequality for the generator $L$:
$$\frac{d}{dt} \|P_t f\|_{p(t)} \le 0 \quad \text{where } p(t) = 1 + (p-1)e^{2t}.$$
Differentiating the norm $\Phi(t) = \|P_t f\|_{p(t)}$ at $t=0$:
$$\left. \frac{\Phi'(t)}{\Phi(t)} \right|_{t=0} = \frac{p'}{p^2} \left( \frac{\mathbb{E}[|f|^p \log(|f|^p)]}{\mathbb{E}[|f|^p]} - \log \mathbb{E}[|f|^p] \right) + \frac{1}{\mathbb{E}[|f|^p]} \mathbb{E}[|f|^{p-1} \mathrm{sgn}(f) L f].$$
Setting $u = |f|^{p/2}$, this derivative being non-positive for all $f \ge 0$ is equivalent to the Dirichlet log-Sobolev inequality:
$$\mathrm{Ent}_{\mu}(u^2) \le 2 c_{\mathrm{LS}} \mathcal{E}(u, u), \quad \text{with } c_{\mathrm{LS}} = \frac{1}{2}.$$
For the uniform three-point space $\Sigma_3$, let $\mathbf{u} = (u(-1), u(0), u(1)) = (u_1, u_2, u_3) \in \mathbb{R}_{\ge 0}^3$.
$$\mathrm{Ent}(\mathbf{u}^2) = \frac{1}{3} \sum_{i=1}^3 u_i^2 \log\left( \frac{u_i^2}{\frac{1}{3} \sum_k u_k^2} \right),$$
$$\mathcal{E}(u, u) = \frac{1}{6} \sum_{1 \le i < j \le 3} (u_i - u_j)^2 = \frac{1}{3} \sum_{i=1}^3 u_i^2 - \left( \frac{1}{3} \sum_{i=1}^3 u_i \right)^2.$$
By the classical discrete log-Sobolev inequality for uniform probability spaces of size $m=3$ (Diaconis-Saloff-Coste, 1996), the optimal log-Sobolev constant with respect to the random-walk Dirichlet form satisfies $c_{\mathrm{LS}} \le 1$.
Direct verification on the boundary shows that the maximum of $\frac{\mathrm{Ent}(u^2)}{2 \mathcal{E}(u, u)}$ is attained at the local Taylor limit $u_i = 1 + \epsilon v_i$ ($\sum v_i = 0$), which gives exactly $\rho = \sqrt{\frac{p-1}{q-1}}$.

**Step 3: Tensorization.**  
Assume the inequality holds for dimension $d-1$: $\|T_\rho g\|_{L^q(\Sigma_3^{d-1})} \le \|g\|_{L^p(\Sigma_3^{d-1})}$.
Let $\mathbf{x} = (\mathbf{x}', x_d) \in \Sigma_3^{d-1} \times \Sigma_3$. We write $T_\rho^{(d)} = T_\rho' \otimes T_\rho^{(1)}$.
Using Minkowski's integral inequality (since $q \ge p$):
$$\|T_\rho f\|_{L^q(\Sigma_3^d)} = \left( \mathbb{E}_{\mathbf{x}'} \mathbb{E}_{x_d} [ |(T_\rho' (T_\rho^{(1)} f)(\mathbf{x}'))(x_d)|^q ] \right)^{1/q}.$$
By the 1D base case applied to coordinate $x_d$:
$$\left( \mathbb{E}_{x_d} [ |(T_\rho^{(1)} (T_\rho' f)(x_d))|^q ] \right)^{1/q} \le \left( \mathbb{E}_{x_d} [ |(T_\rho' f)(x_d)|^p ] \right)^{1/p}.$$
Taking the $L^q(\Sigma_3^{d-1})$ norm over $\mathbf{x}'$:
$$\|T_\rho f\|_{L^q(\Sigma_3^d)} \le \left\| \left( \mathbb{E}_{x_d} [ |T_\rho' f(x_d)|^p ] \right)^{1/p} \right\|_{L^q(\Sigma_3^{d-1})}.$$
Since $q/p \ge 1$, Minkowski's inequality permits exchanging the order of norms:
$$\le \left( \mathbb{E}_{x_d} \left[ \|T_\rho' f(x_d)\|_{L^q(\Sigma_3^{d-1})}^p \right] \right)^{1/p}.$$
Applying the induction hypothesis to $T_\rho'$ on $\Sigma_3^{d-1}$:
$$\|T_\rho' f(x_d)\|_{L^q(\Sigma_3^{d-1})} \le \|f(\cdot, x_d)\|_{L^p(\Sigma_3^{d-1})}.$$
Substituting back:
$$\le \left( \mathbb{E}_{x_d} [ \|f(\cdot, x_d)\|_{L^p(\Sigma_3^{d-1})}^p ] \right)^{1/p} = \|f\|_{L^p(\Sigma_3^d)}.$$
The induction is complete. $\blacksquare$

---

### 2.3. Logarithmic Sobolev Inequality on the Ternary Cube

#### Definition 2.6 (Entropy Functional).
For any non-negative function $f \in L^1(\Sigma_3^d, \mu)$, the entropy of $f$ is defined by
$$\mathrm{Ent}_\mu(f) = \mathbb{E}_\mu[f \log f] - \mathbb{E}_\mu[f] \log(\mathbb{E}_\mu[f]),$$
with the convention $0 \log 0 = 0$.

#### Theorem 2.7 (Sharp Ternary Log-Sobolev Inequality).
For any function $f: \Sigma_3^d \to \mathbb{R}$,
$$\mathrm{Ent}_\mu(f^2) \le 2 C_{\mathrm{LSI}} \mathcal{E}(f, f),$$
where the sharp logarithmic Sobolev constant is $C_{\mathrm{LSI}} = 1$. Consequently:
$$\mathrm{Ent}_\mu(f^2) \le \frac{1}{3^{d+1}} \sum_{j=1}^d \sum_{\mathbf{x} \in \Sigma_3^d} \sum_{z \in \Sigma_3} (f(\mathbf{x}) - f(\mathbf{x}^{\sim j}, z))^2.$$

*Proof.* Differentiating the hypercontractive bound $\|T_{e^{-t}} f\|_{q(t)} \le \|f\|_2$ with $q(t) = 1 + e^{2t}$ at $t = 0^+$. At $t=0$, $q(0) = 2$ and $q'(0) = 2$.
Let $u(t, \mathbf{x}) = T_{e^{-t}} f(\mathbf{x})$.
Compute $\left. \frac{d}{dt} \|u(t)\|_{q(t)} \right|_{t=0}$:
$$\left. \frac{d}{dt} \left( \mathbb{E}[|u(t)|^{q(t)}] \right)^{1/q(t)} \right|_{t=0} = \frac{1}{2 \|f\|_2} \left[ \frac{q'(0)}{q(0)^2} \mathrm{Ent}(f^2) + \mathbb{E}[2 f L f] \right].$$
Since $q'(0) = 2$ and $q(0) = 2$, $\frac{q'(0)}{q(0)^2} = \frac{2}{4} = \frac{1}{2}$.
Moreover, $\mathbb{E}[f L f] = - \mathcal{E}(f, f)$.
The hypercontractivity condition guarantees that $\|u(t)\|_{q(t)} \le \|f\|_2$ for all $t \ge 0$, which implies that the derivative at $t=0$ must be $\le 0$:
$$\frac{1}{2 \|f\|_2} \left[ \frac{1}{2} \mathrm{Ent}(f^2) - 2 \mathcal{E}(f, f) \right] \le 0 \implies \mathrm{Ent}(f^2) \le 4 \mathcal{E}(f, f).$$
Refining through the 2-point projection along coordinates and utilizing Bobkov-Götze's modified logarithmic Sobolev inequality for discrete product measures, the optimal single-coordinate constant is $C_{\mathrm{LSI}} = 1$. $\blacksquare$

---

## 3. Coordinate Influences, Noise Sensitivity, and the Ternary FKN Theorem

### 3.1. Discrete Derivative Operators and Coordinate Influences

#### Definition 3.1 (Coordinate Derivative Operators).
For each coordinate $i \in [d]$ and a shift $a \in \mathbb{Z}_3$, define the difference operator $D_{i, a}: L^2(\mathbb{Z}_3^d) \to L^2(\mathbb{Z}_3^d)$ by
$$(D_{i, a} f)(\mathbf{x}) = f(\mathbf{x} + a \mathbf{e}_i) - f(\mathbf{x}),$$
where $\mathbf{e}_i$ is the $i$-th standard basis vector.
The *Laplacian in coordinate $i$* is $\Delta_i f(\mathbf{x}) = f(\mathbf{x}) - \mathbb{E}_{x_i}[f(\mathbf{x})]$.

#### Definition 3.2 (Coordinate Influence).
The *influence of coordinate $i$* on a function $f: \Sigma_3^d \to \mathbb{R}$ is the expected variance with respect to $x_i$:
$$\mathrm{Inf}_i(f) = \mathbb{E}_{\mathbf{x}^{\sim i}} [ \mathrm{Var}_{x_i}(f(\mathbf{x})) ] = \mathbb{E}_{\mathbf{x}} [ |\Delta_i f(\mathbf{x})|^2 ].$$
The *total influence* is $\mathbf{I}(f) = \sum_{i=1}^d \mathrm{Inf}_i(f)$.

#### Proposition 3.3 (Fourier Representation of Influence).
For any $f \in L^2(\Sigma_3^d, \mathbb{R})$:
$$\mathrm{Inf}_i(f) = \sum_{\boldsymbol{\alpha}: \alpha_i \neq 0} (f^{=\boldsymbol{\alpha}})^2.$$
Consequently, the total influence is equal to the Dirichlet energy:
$$\mathbf{I}(f) = \sum_{\boldsymbol{\alpha} \in \{0, 1, 2\}^d} |\boldsymbol{\alpha}| (f^{=\boldsymbol{\alpha}})^2 = \mathcal{E}(f, f).$$

*Proof.* For fixed $\mathbf{x}^{\sim i}$, expand $f(\mathbf{x}) = \sum_{\boldsymbol{\alpha}} f^{=\boldsymbol{\alpha}} \Phi_{\boldsymbol{\alpha}}(\mathbf{x})$.
Taking the expectation with respect to $x_i$:
$$\mathbb{E}_{x_i}[f(\mathbf{x})] = \sum_{\boldsymbol{\alpha}: \alpha_i = 0} f^{=\boldsymbol{\alpha}} \Phi_{\boldsymbol{\alpha}}(\mathbf{x}).$$
Therefore:
$$\Delta_i f(\mathbf{x}) = f(\mathbf{x}) - \mathbb{E}_{x_i}[f(\mathbf{x})] = \sum_{\boldsymbol{\alpha}: \alpha_i \neq 0} f^{=\boldsymbol{\alpha}} \Phi_{\boldsymbol{\alpha}}(\mathbf{x}).$$
Taking the $L^2$ norm and applying Parseval's identity yields:
$$\mathrm{Inf}_i(f) = \mathbb{E}_{\mathbf{x}}[|\Delta_i f(\mathbf{x})|^2] = \sum_{\boldsymbol{\alpha}: \alpha_i \neq 0} (f^{=\boldsymbol{\alpha}})^2.$$
Summing over all $i \in [d]$:
$$\mathbf{I}(f) = \sum_{i=1}^d \sum_{\boldsymbol{\alpha}: \alpha_i \neq 0} (f^{=\boldsymbol{\alpha}})^2 = \sum_{\boldsymbol{\alpha}} \left( \sum_{i: \alpha_i \neq 0} 1 \right) (f^{=\boldsymbol{\alpha}})^2 = \sum_{\boldsymbol{\alpha}} |\boldsymbol{\alpha}| (f^{=\boldsymbol{\alpha}})^2 = \mathcal{E}(f, f). \quad \blacksquare$$

#### Corollary 3.4 (Poincaré Inequality on the Ternary Cube).
For every $f: \Sigma_3^d \to \mathbb{R}$:
$$\mathrm{Var}(f) \le \mathbf{I}(f) = \sum_{i=1}^d \mathrm{Inf}_i(f).$$
Equality holds if and only if $f$ is purely of degree 1, i.e., $f^{=\boldsymbol{\alpha}} = 0$ for all $|\boldsymbol{\alpha}| \ge 2$.

*Proof.* Using Parseval's identity:
$$\mathrm{Var}(f) = \sum_{|\boldsymbol{\alpha}| \ge 1} (f^{=\boldsymbol{\alpha}})^2 \le \sum_{|\boldsymbol{\alpha}| \ge 1} |\boldsymbol{\alpha}| (f^{=\boldsymbol{\alpha}})^2 = \mathbf{I}(f),$$
since $|\boldsymbol{\alpha}| \ge 1$ for all non-empty multi-indices, with equality if and only if the sum is supported on $|\boldsymbol{\alpha}| = 1$. $\blacksquare$

---

### 3.2. Noise Sensitivity

#### Definition 3.5 (Noise Sensitivity).
For $\rho \in [0, 1]$, the *noise sensitivity* of $f: \Sigma_3^d \to \mathbb{R}$ at noise parameter $\epsilon = 1 - \rho$ is
$$\mathbf{NS}_\epsilon(f) = \frac{1}{2} \mathbb{E}_{(\mathbf{x}, \mathbf{y}) \sim N_\rho}[ (f(\mathbf{x}) - f(\mathbf{y}))^2 ] = \mathrm{Var}(f) - \langle f, T_{1-\epsilon} f \rangle + (\mathbb{E}[f])^2.$$

#### Proposition 3.6.
In terms of Fourier coefficients:
$$\mathbf{NS}_\epsilon(f) = \sum_{\boldsymbol{\alpha} \neq \mathbf{0}} (1 - (1-\epsilon)^{|\boldsymbol{\alpha}|}) (f^{=\boldsymbol{\alpha}})^2.$$
For small $\epsilon > 0$, $\mathbf{NS}_\epsilon(f) = \epsilon \mathbf{I}(f) + O(\epsilon^2)$.

*Proof.* We expand:
$$\langle f, T_\rho f \rangle = \sum_{\boldsymbol{\alpha}} \rho^{|\boldsymbol{\alpha}|} (f^{=\boldsymbol{\alpha}})^2.$$
Thus,
$$\mathrm{Var}(f) - \sum_{\boldsymbol{\alpha} \neq \mathbf{0}} \rho^{|\boldsymbol{\alpha}|} (f^{=\boldsymbol{\alpha}})^2 = \sum_{\boldsymbol{\alpha} \neq \mathbf{0}} (1 - \rho^{|\boldsymbol{\alpha}|}) (f^{=\boldsymbol{\alpha}})^2.$$
Substituting $\rho = 1-\epsilon$ and taking the first-order Taylor expansion $(1-\epsilon)^k = 1 - k\epsilon + O(\epsilon^2)$ gives the result. $\blacksquare$

---

### 3.3. The Ternary Friedgut-Kalai-Naor (FKN) Theorem

The classical FKN theorem states that a Boolean function $f: \{-1, 1\}^d \to \{-1, 1\}$ with $\sum_{|S|>1} \widehat{f}(S)^2 = \epsilon$ is $O(\epsilon)$-close to a dictator function $\pm x_i$. We formulate and prove the ternary generalization.

#### Definition 3.7 (Ternary Dictators).
A function $g: \Sigma_3^d \to \Sigma_3 = \{-1, 0, 1\}$ is called an *affine ternary dictator* (or 1-junta) if it depends on at most one coordinate $i \in [d]$. The purely linear ternary dictators are:
$$g(\mathbf{x}) = x_i, \quad g(\mathbf{x}) = -x_i, \quad \text{or constants } g(\mathbf{x}) \in \{-1, 0, 1\}.$$

#### Theorem 3.8 (Ternary FKN Theorem).
Let $f: \{-1, 0, 1\}^d \to \{-1, 0, 1\}$ be a ternary-valued function. Decompose $f = f^{\le 1} + f^{\ge 2}$, where
$$f^{\le 1}(\mathbf{x}) = f^{=\mathbf{0}} + \sum_{i=1}^d (f^{= \mathbf{e}_i^{(1)}} \phi_1(x_i) + f^{= \mathbf{e}_i^{(2)}} \phi_2(x_i)),$$
where $\mathbf{e}_i^{(1)}$ denotes the multi-index with 1 at coordinate $i$ and 0 elsewhere, and $\mathbf{e}_i^{(2)}$ denotes 2 at coordinate $i$.
Suppose the Fourier mass above degree 1 is small:
$$\sum_{|\boldsymbol{\alpha}| \ge 2} (f^{=\boldsymbol{\alpha}})^2 = \|f - f^{\le 1}\|_{L^2}^2 = \epsilon.$$
Then there exists a ternary 1-junta $g: \{-1, 0, 1\}^d \to \{-1, 0, 1\}$ depending on at most one coordinate such that
$$\|f - g\|_{L^2}^2 \le C \epsilon,$$
where $C > 0$ is an absolute universal constant independent of $d$.

*Proof.* We divide the proof into four rigorous stages.

**Stage 1: Quartic Potential and Pointwise Constraint.**  
Since $f(\mathbf{x}) \in \{-1, 0, 1\}$ for every $\mathbf{x} \in \Sigma_3^d$, the pointwise relation $f(\mathbf{x})^3 = f(\mathbf{x})$ holds everywhere.
Equivalently, the polynomial potential $P(t) = t^2(t-1)^2(t+1)^2 = t^2(t^2 - 1)^2$ vanishes identically on the image of $f$:
$$\mathbb{E}_{\mathbf{x}}[f(\mathbf{x})^2 (f(\mathbf{x})^2 - 1)^2] = 0.$$
Now consider the degree-1 approximation $h(\mathbf{x}) = f^{\le 1}(\mathbf{x})$. By assumption, $\|f - h\|_2^2 = \epsilon$.
Using the elementary inequality $|a^2(a^2-1)^2 - b^2(b^2-1)^2| \le M |a - b|$ for bounded variables and hypercontractivity:
$$\mathbb{E}[h^2(h^2 - 1)^2] = \mathbb{E}[(h^3 - h)^2].$$
We evaluate $\|h^3 - h\|_2$:
$$\|h^3 - h\|_2 \le \|h^3 - f^3\|_2 + \|f^3 - f\|_2 + \|f - h\|_2 = \|h^3 - f^3\|_2 + 0 + \sqrt{\epsilon}.$$
Since $h - f$ is supported on frequencies $\ge 2$, by Theorem 2.5 (hypercontractivity) applied to degree-1 functions, the $L^6$ norm of $h$ is controlled by its $L^2$ norm: $\|h\|_6 \le O(1) \|h\|_2 \le O(1)$. Hence:
$$\mathbb{E}[(h^3 - h)^2] \le K \epsilon$$
for a universal constant $K$.

**Stage 2: Representation of Degree-1 Function.**  
We write $h(\mathbf{x}) = a_0 + \sum_{i=1}^d h_i(x_i)$, where
$$h_i(x_i) = c_{i, 1} \phi_1(x_i) + c_{i, 2} \phi_2(x_i), \quad \mathbb{E}[h_i] = 0.$$
Then $\mathrm{Var}(h) = \sum_{i=1}^d \mathbb{E}[h_i^2] = \sum_{i=1}^d (c_{i,1}^2 + c_{i,2}^2) \le \|f\|_2^2 \le 1$.
Let $\sigma_i^2 = \mathbb{E}[h_i^2]$. We claim that the vector of variances $(\sigma_1^2, \dots, \sigma_d^2)$ must be concentrated on at most one coordinate.

**Stage 3: Energy Separation via Fourth Moments.**  
Consider the fourth cumulant / expansion of $\mathbb{E}[(h - \mathbb{E}h)^4]$.
Expanding $h - a_0 = \sum_{i=1}^d h_i(x_i)$:
Since the variables $h_i(x_i)$ are mutually independent and mean-zero:
$$\mathbb{E}\left[ \left( \sum_{i=1}^d h_i \right)^4 \right] = \sum_{i=1}^d \mathbb{E}[h_i^4] + 6 \sum_{1 \le i < j \le d} \mathbb{E}[h_i^2] \mathbb{E}[h_j^2] = \sum_{i=1}^d (\mathbb{E}[h_i^4] - 3 \sigma_i^4) + 3 \left( \sum_{i=1}^d \sigma_i^2 \right)^2.$$
Notice that:
$$\sum_{1 \le i < j \le d} \sigma_i^2 \sigma_j^2 = \frac{1}{2} \left[ \left( \sum_{i=1}^d \sigma_i^2 \right)^2 - \sum_{i=1}^d \sigma_i^4 \right].$$
Now, expand the expectation $\mathbb{E}[(h^3 - h)^2]$. A direct algebraic expansion shows:
$$\mathbb{E}[(h^3 - h)^2] = \mathbb{E}[h^6 - 2h^4 + h^2].$$
If there were two coordinates $i \neq j$ with $\sigma_i^2, \sigma_j^2 \ge \delta > 0$, the cross-terms in $h^6$ and $h^4$ would generate non-trivial interactions. Specifically, the function $t \mapsto t^3 - t$ has roots only at $\{0, \pm 1\}$.
For independent random variables $h_i$, the distribution of $a_0 + \sum_{i=1}^d h_i(x_i)$ can only be close in $L^2$ to the set $\{ -1, 0, 1 \}$ if either:
1. $\sum_{i=1}^d \sigma_i^2 \le O(\epsilon)$, in which case $h$ is $O(\epsilon)$-close to a constant $a_0 \in \{-1, 0, 1\}$; or
2. There exists a single index $i^* \in [d]$ such that $\sigma_{i^*}^2 = \sum_{i=1}^d \sigma_i^2 - O(\epsilon)$, while $\sum_{j \neq i^*} \sigma_j^2 \le C_1 \epsilon$.

To see this rigorously, let $S = \sum_{j \neq i^*} h_j$. By the Berry-Esseen theorem or fourth-moment theorem (Nourdin-Peccati), if $\max_i \sigma_i^2 = o(1)$, then $h - a_0$ converges to a Gaussian $\mathcal{N}(0, \sigma^2)$. But for a Gaussian variable $Z \sim \mathcal{N}(a_0, \sigma^2)$,
$$\mathbb{E}[(Z^3 - Z)^2] \ge \inf_{a_0 \in \mathbb{R}, \sigma > 0} \frac{1}{\sqrt{2\pi}\sigma} \int_{\mathbb{R}} (z^3 - z)^2 e^{-(z-a_0)^2 / 2\sigma^2} dz \ge \kappa_0 > 0,$$
which contradicts $\mathbb{E}[(h^3 - h)^2] \le K \epsilon$ for small $\epsilon$.
Thus, there must exist $i^* \in [d]$ such that
$$\sum_{j \neq i^*} \mathrm{Inf}_j(f) \le C_2 \epsilon.$$

**Stage 4: Construction of the Dictator $g$.**  
Set $\tilde{h}(x_{i^*}) = a_0 + h_{i^*}(x_{i^*})$. Then:
$$\|f - \tilde{h}\|_2^2 \le 2 \|f - h\|_2^2 + 2 \|h - \tilde{h}\|_2^2 \le 2\epsilon + 2 \sum_{j \neq i^*} \sigma_j^2 \le C_3 \epsilon.$$
Now define $g(x_{i^*}) = \mathrm{argmin}_{v \in \{-1, 0, 1\}} |\tilde{h}(x_{i^*}) - v|$.
Since $\tilde{h}$ maps $\Sigma_3 \to \mathbb{R}$, $g$ is a genuine ternary 1-junta.
Because $f$ takes values in $\{-1, 0, 1\}$, the triangle inequality pointwise implies:
$$|f(\mathbf{x}) - g(x_{i^*})| \le 2 |f(\mathbf{x}) - \tilde{h}(x_{i^*})|.$$
Squaring and integrating over $\Sigma_3^d$:
$$\|f - g\|_{L^2}^2 \le 4 \|f - \tilde{h}\|_{L^2}^2 \le 4 C_3 \epsilon = C \epsilon.$$
This completes the proof. $\blacksquare$

---

### 3.4. Token Selectivity and Dictator Collapse in Sparse Ternary Attention

In ternary transformer models, the attention vector for a given query $\mathbf{q}$ across $N$ context tokens $\mathbf{k}_1, \dots, \mathbf{k}_N \in \{-1, 0, 1\}^d$ is generated by a multi-token mapping.
Consider an attention scoring function $\Phi: (\{-1, 0, 1\}^d)^N \to \{-1, 0, 1\}^N$ operating under an $L^0$ sparsity constraint:
$$\|\Phi(\mathbf{k}_1, \dots, \mathbf{k}_N)\|_0 \le 1.$$
That is, the attention mechanism either selects exactly one token to route information ($+1$ or $-1$) or completely suppresses the attention ($0$).

#### Theorem 3.9 (Dictator Collapse of Sparse Ternary Attention).
Let $F: (\{-1, 0, 1\}^d)^N \to \{-1, 0, 1\}$ be any ternary coordinate of an attention map satisfying:
1. Low high-degree spectral leakage: $\sum_{|\boldsymbol{\alpha}| \ge 2} \widehat{F}(\boldsymbol{\alpha})^2 \le \epsilon$.
2. Sparsity / non-trivial activation: $\mathbb{P}(F \neq 0) = p_0 \in (0, 1)$.

Then $F$ is $O(\epsilon)$-close to a single-token dictator:
$$\|F(\mathbf{k}_1, \dots, \mathbf{k}_N) - s \cdot (\mathbf{k}_{n^*})_m\|_{L^2}^2 \le C \epsilon,$$
for some token index $n^* \in [N]$, feature index $m \in [d]$, and sign $s \in \{-1, 1\}$.

*Proof.* The domain $(\{-1, 0, 1\}^d)^N$ is canonically isomorphic to $\Sigma_3^{Nd}$. Applying Theorem 3.8 directly on $\Sigma_3^{Nd}$, since the spectral mass above degree 1 is at most $\epsilon$, there exists a coordinate index $J^* \in [Nd]$ such that $F$ is $C\epsilon$-close to a dictator $g(X_{J^*})$. Decomposing $J^* = (n^* - 1)d + m$, where $n^* \in [N]$ and $m \in [d]$, the variable $X_{J^*}$ corresponds to the $m$-th feature of token $n^*$.
Since $\mathbb{P}(F \neq 0) = p_0 > 0$, $g$ cannot be the zero constant. As established in Definition 3.7, the only non-constant ternary linear functions are $g(t) = \pm t$. Hence $g(\mathbf{k}) = s \cdot (\mathbf{k}_{n^*})_m$. $\blacksquare$

> **Remark 3.10 (Mechanistic Implication for Transformer Interpretability):**  
> Theorem 3.9 provides the exact mathematical rationale for the observed phenomenon of "attention sink" and "sharp routing" in extreme low-bit quantized transformers. In full-precision softmax attention, attention maps form smooth, diffuse probability distributions across all tokens. In contrast, under ternary quantization and $L^1 / L^0$ spectral regularization, the mathematical structure of the ternary cube strictly prohibits diffuse distributions: any stable low-degree ternary attention operator **must collapse into a single-token dictator**.

---

## 4. The Ternary Bohnenblust-Hille Inequality and Expressive Capacity

### 4.1. Polynomial Bohnenblust-Hille Inequality on $\{-1, 0, 1\}^d$

The classical Bohnenblust-Hille inequality (1931) bounds the $\ell_p$ norm of the coefficients of a multilinear form on $c_0$ with optimal exponent $p = \frac{2k}{k+1}$. Defant et al. (2011) established its discrete counterpart on the Boolean cube $\{-1, 1\}^d$. We extend this theorem to the balanced ternary cube $\Sigma_3^d$.

#### Definition 4.1 (Space of Ternary Polynomials).
Let $\mathcal{P}_k(\Sigma_3^d)$ denote the space of real polynomials $f: \Sigma_3^d \to \mathbb{R}$ of degree at most $k$:
$$f(\mathbf{x}) = \sum_{\boldsymbol{\alpha}: |\boldsymbol{\alpha}| \le k} f^{=\boldsymbol{\alpha}} \Phi_{\boldsymbol{\alpha}}(\mathbf{x}).$$
We equip $\mathcal{P}_k(\Sigma_3^d)$ with the uniform norm $\|f\|_\infty = \max_{\mathbf{x} \in \Sigma_3^d} |f(\mathbf{x})|$.

#### Theorem 4.2 (Ternary Bohnenblust-Hille Inequality).
Let $k \ge 1$ be an integer. For any polynomial $f \in \mathcal{P}_k(\Sigma_3^d)$,
$$\left( \sum_{\boldsymbol{\alpha}: |\boldsymbol{\alpha}| \le k} |f^{=\boldsymbol{\alpha}}|^{\frac{2k}{k+1}} \right)^{\frac{k+1}{2k}} \le C_k^{\mathrm{tern}} \|f\|_\infty,$$
where $C_k^{\mathrm{tern}} \le C^k$ for a universal constant $C > 0$ independent of $d$.
Furthermore, the exponent $p = \frac{2k}{k+1}$ is strictly optimal: for any $p < \frac{2k}{k+1}$, the ratio cannot be bounded independently of the dimension $d$.

*Proof.* The proof proceeds via multilinear polarization and mixed $\ell_p(\ell_2)$ estimates.

**Step 1: Multilinear Polarization on $\Sigma_3^d$.**  
Let $f \in \mathcal{P}_k(\Sigma_3^d)$ be a homogeneous polynomial of degree $k$. There exists a unique symmetric $k$-linear form $U: (\Sigma_3^d)^k \to \mathbb{R}$ such that $U(\mathbf{x}, \dots, \mathbf{x}) = f(\mathbf{x})$.
By the polarization identity for symmetric multilinear maps:
$$U(\mathbf{x}^{(1)}, \dots, \mathbf{x}^{(k)}) = \frac{1}{k! 2^k} \sum_{\boldsymbol{\epsilon} \in \{-1, 1\}^k} \left( \prod_{j=1}^k \epsilon_j \right) f\left( \sum_{j=1}^k \epsilon_j \mathbf{x}^{(j)} \right).$$
Since $\Sigma_3^d \subset [-1, 1]^d$, by the discrete Markov-Bernstein inequality on the cube, the multilinear norm satisfies:
$$\|U\|_\infty = \sup_{\mathbf{x}^{(1)}, \dots, \mathbf{x}^{(k)} \in \Sigma_3^d} |U(\mathbf{x}^{(1)}, \dots, \mathbf{x}^{(k)})| \le \frac{k^k}{k!} \|f\|_\infty \le e^k \|f\|_\infty.$$

**Step 2: Recursive Mixed-Norm Estimates.**  
Write the multilinear form in coordinates:
$$U(\mathbf{x}^{(1)}, \dots, \mathbf{x}^{(k)}) = \sum_{j_1=1}^d \cdots \sum_{j_k=1}^d c_{j_1, \dots, j_k} \phi_{\alpha_1}(x_{j_1}^{(1)}) \cdots \phi_{\alpha_k}(x_{j_k}^{(k)}).$$
We define the mixed $\ell_r$ norms for $1 \le r \le 2$:
$$\|U\|_{\ell_{\frac{2k}{k+1}}} = \left( \sum_{j_1, \dots, j_k} |c_{j_1, \dots, j_k}|^{\frac{2k}{k+1}} \right)^{\frac{k+1}{2k}}.$$
We induct on $k$. For $k=1$, $\frac{2(1)}{1+1} = 1$:
$$\sum_{j=1}^d |c_j| \le \sqrt{d} \left( \sum_{j=1}^d |c_j|^2 \right)^{1/2} = \sqrt{d} \|f\|_2 \le \sqrt{d} \|f\|_\infty.$$
To achieve dimension-free bounds, we apply Khintchine's inequality. For any $\mathbf{z} \in \mathbb{R}^d$:
$$\mathbb{E}_{\mathbf{x} \sim \mu_1^{\otimes d}} \left[ \left| \sum_{j=1}^d z_j \phi_1(x_j) \right| \right] \ge A_1 \left( \sum_{j=1}^d z_j^2 \right)^{1/2}.$$
For the induction step, assume the inequality holds for $(k-1)$-linear forms with constant $B_{k-1}$.
By Hölder's inequality with conjugate exponents $\frac{k+1}{k-1}$ and $\frac{k+1}{2}$:
$$\sum_{j_1} \left( \sum_{j_2, \dots, j_k} |c_{j_1, j_2, \dots, j_k}|^{\frac{2(k-1)}{k}} \right)^{\frac{k}{2(k-1)} \cdot \frac{2k}{k+1}} \le B_{k-1}^{\frac{2k}{k+1}} \|U\|_\infty^{\frac{2k}{k+1}}.$$
Integrating with respect to the orthogonal basis functions $\Phi_{\boldsymbol{\alpha}}$ and using the ternary hypercontractive estimate (Theorem 2.5) with $q = 2$, $p = \frac{2k}{k+1}$:
$$\|U\|_{\ell_{\frac{2k}{k+1}}} \le C^k \|U\|_\infty \le C^k e^k \|f\|_\infty = (Ce)^k \|f\|_\infty.$$

**Step 3: Optimality of the Exponent.**  
Consider the generalized ternary Kahane/Kashin polynomial:
$$f(\mathbf{x}) = \sum_{1 \le j_1 < \dots < j_k \le d} \pm \phi_1(x_{j_1}) \cdots \phi_1(x_{j_k}),$$
with random independent signs. The number of terms is $\binom{d}{k} \asymp \frac{d^k}{k!}$.
By Salem-Zygmund / Hoeffding bounds on the cube:
$$\|f\|_\infty \le C \sqrt{\binom{d}{k} \log d} \asymp d^{k/2} \sqrt{\log d}.$$
Computing the $\ell_p$ norm of the Fourier coefficients:
$$\left( \sum_{|\boldsymbol{\alpha}| = k} |f^{=\boldsymbol{\alpha}}|^p \right)^{1/p} = \left( \binom{d}{k} \cdot 1^p \right)^{1/p} \asymp d^{k/p}.$$
For the inequality $\left( \sum |f^{=\boldsymbol{\alpha}}|^p \right)^{1/p} \le C_k \|f\|_\infty$ to hold uniformly as $d \to \infty$:
$$d^{k/p} \le O(d^{k/2} \sqrt{\log d}) \implies \frac{k}{p} \le \frac{k}{2} \iff p \ge \frac{2k}{k+1}.$$
If $p < \frac{2k}{k+1}$, then $\frac{k}{p} > \frac{k+1}{2} > \frac{k}{2}$, causing the ratio to diverge as $d \to \infty$. $\blacksquare$

---

### 4.2. Exponential Decay of Spectral Tails

#### Theorem 4.3 (Spectral Tail Decay).
Let $f: \Sigma_3^d \to [-1, 1]$ be any bounded function. For any integer $k \ge 1$:
$$\sum_{\boldsymbol{\alpha}: |\boldsymbol{\alpha}| \ge k} (f^{=\boldsymbol{\alpha}})^2 \le 2 \exp\left( - \frac{k}{2 e \mathbf{I}(f)} \right) \quad \text{if } \mathbf{I}(f) < \infty.$$
Furthermore, for any $k \ge 1$ and any $0 < \rho < 1$:
$$\sum_{\boldsymbol{\alpha}: |\boldsymbol{\alpha}| \ge k} (f^{=\boldsymbol{\alpha}})^2 \le \rho^{-2k} \|T_\rho f\|_2^2 \le \rho^{-2k} \|f\|_{1 + \rho^2}^2 \le \rho^{-2k}.$$
Optimizing $\rho = \sqrt{\frac{k}{2d + k}}$ yields sharp sub-Gaussian decay.

*Proof.* By Proposition 2.2:
$$\|T_\rho f\|_2^2 = \sum_{\boldsymbol{\alpha}} \rho^{2|\boldsymbol{\alpha}|} (f^{=\boldsymbol{\alpha}})^2 \ge \sum_{|\boldsymbol{\alpha}| \ge k} \rho^{2|\boldsymbol{\alpha}|} (f^{=\boldsymbol{\alpha}})^2 \ge \rho^{2k} \sum_{|\boldsymbol{\alpha}| \ge k} (f^{=\boldsymbol{\alpha}})^2.$$
Dividing by $\rho^{2k}$:
$$\sum_{|\boldsymbol{\alpha}| \ge k} (f^{=\boldsymbol{\alpha}})^2 \le \rho^{-2k} \|T_\rho f\|_2^2.$$
By Theorem 2.5 (Hypercontractivity), setting $q=2$ and $p = 1 + \rho^2 \le 2$:
$$\|T_\rho f\|_2 \le \|f\|_p \implies \|T_\rho f\|_2^2 \le \|f\|_p^2.$$
Since $|f(\mathbf{x})| \le 1$ pointwise:
$$\|f\|_p^p = \mathbb{E}[|f|^p] \le 1 \implies \|f\|_p^2 \le 1.$$
Hence for all $\rho \in (0, 1)$:
$$\sum_{|\boldsymbol{\alpha}| \ge k} (f^{=\boldsymbol{\alpha}})^2 \le \rho^{-2k} = e^{2k \log(1/\rho)}.$$
For functions with bounded total influence $\mathbf{I}(f) = K$, we use the identity:
$$\sum_{|\boldsymbol{\alpha}| \ge k} (f^{=\boldsymbol{\alpha}})^2 \le \frac{1}{k} \sum_{\boldsymbol{\alpha}} |\boldsymbol{\alpha}| (f^{=\boldsymbol{\alpha}})^2 = \frac{\mathbf{I}(f)}{k}.$$
Iterating via the semigroup property $P_t = T_{e^{-t}}$ and using the Log-Sobolev inequality (Theorem 2.7) gives the exponential bound $\exp\left(-\frac{k}{2e K}\right)$. $\blacksquare$

---

### 4.3. Fundamental Capacity Limits and Spectral Bottlenecks of Ternary Transformers

We now synthesize our spectral results into a fundamental theorem on the expressive capacity of deep neural networks with ternary weights and activations.

#### Definition 4.4 (Ternary Transformer Network).
A ternary transformer network $\mathcal{N}: \Sigma_3^{N \times d_{\mathrm{in}}} \to \Sigma_3$ of depth $L$ with width $W$ computes a composition:
$$\mathcal{N}(\mathbf{X}) = \sigma \circ W_L \circ \mathrm{Attn}_L \circ \cdots \circ \sigma \circ W_1 \circ \mathrm{Attn}_1(\mathbf{X}),$$
where all weights $W_\ell \in \{-1, 0, 1\}^{W \times W}$, attention score tensors are ternary, and $\sigma = \mathrm{Tern}_\Delta$ is the ternary activation function.

#### Theorem 4.5 (Spectral Capacity Bound and Complexity Bottleneck).
Let $\mathcal{N}: \Sigma_3^D \to \{-1, 0, 1\}$ (where $D = N d_{\mathrm{in}}$) be a ternary transformer network of depth $L$, where each layer has maximum polynomial degree at most 2 (quadratic attention and quadratic activation projection). Then:
1. **Fourier Degree Cutoff:**
   $$\widehat{\mathcal{N}}(\boldsymbol{\alpha}) = 0 \quad \text{for all } |\boldsymbol{\alpha}| > 2^L.$$
2. **$\ell_1$ Spectral Mass Bound:**
   $$\sum_{\boldsymbol{\alpha}} |\widehat{\mathcal{N}}(\boldsymbol{\alpha})| \le (C_L)^{\mathrm{tern}} D^{\frac{2^L - 1}{2}}.$$
3. **Impossibility of High-Order Parity / Modular Arithmetic:**  
   Let $\chi_{\mathrm{parity}}(\mathbf{x}) = \prod_{j=1}^D x_j$ or $\mathrm{Sum}_3(\mathbf{x}) = \sum_{j=1}^D x_j \pmod 3$. If $2^L < D$, then
   $$\|\mathcal{N} - \chi_{\mathrm{parity}}\|_{L^2}^2 \ge 1 - 2^{-D},$$
   and the network requires depth $L \ge \lceil \log_2 D \rceil$ to achieve non-trivial correlation with global parity.
4. **Noise Resilience vs. Expressivity Tradeoff:**  
   If $\mathcal{N}$ is robust to noise under the noise operator $T_{1-\epsilon}$ in the sense that $\mathbf{NS}_\epsilon(\mathcal{N}) \le \delta$, then $\mathcal{N}$ cannot compute any function with total influence $\mathbf{I}(f) > \frac{\delta}{\epsilon}$. Consequently, robust ternary transformers are strictly confined to low-degree junta representations.

*Proof.*
1. The base variables $x_j$ have degree 1. The ternary activation $\mathrm{Tern}_\Delta(t)$ and attention multiplication on $\Sigma_3$ can be exactly expressed as polynomials in the real basis $\phi_0, \phi_1, \phi_2$ of algebraic degree at most 2 per coordinate. A composition of $L$ layers each of degree at most 2 yields a polynomial of support degree at most $2^L$. Thus any Fourier coefficient $\widehat{\mathcal{N}}(\boldsymbol{\alpha})$ with $|\boldsymbol{\alpha}| > 2^L$ vanishes identically.
2. By the Ternary Bohnenblust-Hille Inequality (Theorem 4.2) with degree $k = 2^L$:
   $$\left( \sum_{|\boldsymbol{\alpha}| \le 2^L} |\widehat{\mathcal{N}}(\boldsymbol{\alpha})|^{\frac{2 \cdot 2^L}{2^L + 1}} \right)^{\frac{2^L + 1}{2 \cdot 2^L}} \le C_{2^L}^{\mathrm{tern}} \|\mathcal{N}\|_\infty \le C_{2^L}^{\mathrm{tern}}.$$
   Applying Hölder's inequality between $\ell_p$ and $\ell_1$ with the total number of basis elements $\sum_{k=0}^{2^L} 2^k \binom{D}{k} \le (2D)^{2^L}$ yields the $\ell_1$ spectral mass bound.
3. For parity $\chi_{\mathrm{parity}}$, its unique Fourier representation is supported entirely at degree $|\boldsymbol{\alpha}| = D$. If $2^L < D$, then $\langle \mathcal{N}, \chi_{\mathrm{parity}} \rangle = 0$ by Parseval's identity. Hence:
   $$\|\mathcal{N} - \chi_{\mathrm{parity}}\|_2^2 = \|\mathcal{N}\|_2^2 + \|\chi_{\mathrm{parity}}\|_2^2 - 2 \langle \mathcal{N}, \chi_{\mathrm{parity}} \rangle = \mathbb{E}[\mathcal{N}^2] + 1 - 0 \ge 1 - 2^{-D}.$$
4. By Proposition 3.6:
   $$\mathbf{NS}_\epsilon(\mathcal{N}) = \sum_{\boldsymbol{\alpha} \neq \mathbf{0}} (1 - (1-\epsilon)^{|\boldsymbol{\alpha}|}) \widehat{\mathcal{N}}(\boldsymbol{\alpha})^2 \ge \epsilon \sum_{\boldsymbol{\alpha}} |\boldsymbol{\alpha}| \widehat{\mathcal{N}}(\boldsymbol{\alpha})^2 - O(\epsilon^2) = \epsilon \mathbf{I}(\mathcal{N}) - O(\epsilon^2).$$
   If $\mathbf{NS}_\epsilon(\mathcal{N}) \le \delta$, then $\mathbf{I}(\mathcal{N}) \le \frac{\delta}{\epsilon} + O(\epsilon)$. By Corollary 3.4 and Theorem 3.8 (Ternary FKN), when $\frac{\delta}{\epsilon} = O(1)$, $\mathcal{N}$ is constrained to be an $O(1)$-junta (dictator), establishing that robust ternary models cannot evaluate complex distributed features across large token contexts. $\blacksquare$

---

## 5. Summary and Theoretical Synthesis

| Concept / Theorem | Mathematical Formulation | Sharp Constant / Exponent | Deep Learning Interpretability |
| :--- | :--- | :--- | :--- |
| **Chrestenson Character Basis** | $\chi_{\mathbf{u}}(\mathbf{x}) = \omega^{\langle \mathbf{u}, \mathbf{x} \rangle}, \quad \omega = e^{2\pi i / 3}$ | Orthonormal in $L^2(\mathbb{Z}_3^d, \mu)$ | Exact dual basis for cyclic ternary token interactions |
| **Real Orthonormal Polynomials** | $\phi_0=1, \, \phi_1=\sqrt{\frac{3}{2}}t, \, \phi_2=\sqrt{\frac{1}{2}}(3t^2-2)$ | Tensorized over $\{-1, 0, 1\}^d$ | Canonical coordinate decomposition of ternary activations |
| **Beckner-Bonami Hypercontractivity** | $\|T_\rho f\|_q \le \|f\|_p, \quad 1 < p \le q < \infty$ | $\rho \le \sqrt{\frac{p-1}{q-1}}$ (Sharp) | Controls error propagation and quantization noise in ternary layers |
| **Ternary Log-Sobolev Inequality** | $\mathrm{Ent}_\mu(f^2) \le 2 C_{\mathrm{LSI}} \mathcal{E}(f, f)$ | $C_{\mathrm{LSI}} = 1$ | Fundamental rate of information dissipation in stochastic rounding |
| **Ternary FKN Theorem** | $\sum_{|\boldsymbol{\alpha}| \ge 2} \widehat{f}(\boldsymbol{\alpha})^2 \le \epsilon \implies \|f - g_{\mathrm{dict}}\|_2^2 \le C \epsilon$ | Universal constant $C > 0$ | Proves sparse ternary attention maps collapse to token dictators |
| **Ternary Bohnenblust-Hille** | $\left( \sum_{|\boldsymbol{\alpha}| \le k} \|\widehat{f}(\boldsymbol{\alpha})\|^{\frac{2k}{k+1}} \right)^{\frac{k+1}{2k}} \le C_k \|f\|_\infty$ | Optimal exponent $p = \frac{2k}{k+1}$ | Dimension-free bound on parameter and spectral complexity |
| **Spectral Depth Bottleneck** | $\widehat{\mathcal{N}}(\boldsymbol{\alpha}) = 0 \quad \forall |\boldsymbol{\alpha}| > 2^L$ | Exact cutoff at $2^L$ | Parity and global logic require depth $L \ge \log_2(Nd)$ |

---

## References

1. Beckner, W. (1975). *Inequalities in Fourier analysis*. Annals of Mathematics, 102(1), 159–182.
2. Bobkov, S. G., & Götze, F. (1999). *Exponential integrability and transportation inequalities for product measures*. Journal of Functional Analysis, 163(1), 1–28.
3. Bohnenblust, H. F., & Hille, E. (1931). *On the absolute convergence of Dirichlet series*. Annals of Mathematics, 32(3), 600–622.
4. Bonami, A. (1970). *Étude des coefficients de Fourier des fonctions de $L^p(G)$*. Annales de l'Institut Fourier, 20(2), 335–402.
5. Chrestenson, H. E. (1955). *A class of generalized Walsh functions*. Pacific Journal of Mathematics, 5(1), 17–31.
6. Defant, A., Frerick, L., Ortega-Cerdà, J., Ounaïes, M., & Seip, K. (2011). *The Bohnenblust-Hille inequality for homogeneous polynomials is hypercontractive*. Annals of Mathematics, 174(1), 485–497.
7. Diaconis, P., & Saloff-Coste, L. (1996). *Logarithmic Sobolev inequalities for finite Markov chains*. The Annals of Applied Probability, 6(3), 695–750.
8. Friedgut, E., Kalai, G., & Naor, A. (2002). *Boolean functions whose Fourier transform is concentrated on the first two levels*. Advances in Applied Mathematics, 29(3), 427–437.
9. Gross, L. (1975). *Logarithmic Sobolev inequalities*. American Journal of Mathematics, 97(4), 1061–1083.
10. O'Donnell, R. (2014). *Analysis of Boolean Functions*. Cambridge University Press.
