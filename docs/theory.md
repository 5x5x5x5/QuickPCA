# Theory

This page explains the methods QuickPCA implements: essential dynamics, Kabsch
alignment, SVD-based PCA, residue cross-correlation, and the Free-Energy
Landscape via Boltzmann inversion. The notation matches the
[source code](api.md): \(F\) is the number of frames, \(N\) the number of
selected atoms, and \(D = 3N\) the number of Cartesian coordinates per frame.

## Essential dynamics

A molecular-dynamics trajectory is a sequence of \(F\) conformations, each a
point in a \(3N\)-dimensional configuration space. Most of the variance in this
cloud is concentrated in a small number of **collective coordinates** — the
*essential subspace* — that describe large-amplitude, functionally relevant
motions. The remaining directions are dominated by uncorrelated thermal
fluctuations.

Essential Dynamics Analysis (Amadei *et al.*, 1993) identifies this subspace by
diagonalising the covariance matrix of atomic displacements. QuickPCA computes
exactly the same modes, but takes a more efficient and numerically stable route
through the singular value decomposition.

## Kabsch alignment

Before any covariance is meaningful, the overall translation and rotation of the
molecule must be removed so that only **internal** motion remains. QuickPCA uses
the **Kabsch algorithm** to superimpose every frame onto a reference structure
\(\mathbf{R}\) (frame `ref_index`, the first frame by default).

For a frame with coordinates \(\mathbf{P} \in \mathbb{R}^{N\times 3}\):

1. Centre both structures on their centroids,
   \(\tilde{\mathbf{P}} = \mathbf{P} - \bar{\mathbf{P}}\) and
   \(\tilde{\mathbf{R}} = \mathbf{R} - \bar{\mathbf{R}}\).
2. Form the \(3\times 3\) cross-covariance matrix

    \[
    \mathbf{H} = \tilde{\mathbf{P}}^{\mathsf{T}}\,\tilde{\mathbf{R}}.
    \]

3. Take its SVD, \(\mathbf{H} = \mathbf{U}\,\boldsymbol{\Sigma}\,\mathbf{V}^{\mathsf{T}}\),
   and build the optimal rotation while **correcting for reflections**:

    \[
    d = \operatorname{sign}\!\big(\det(\mathbf{V}\mathbf{U}^{\mathsf{T}})\big),
    \qquad
    \mathbf{R}_{\mathrm{rot}} = \mathbf{V}
        \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & d \end{pmatrix}
        \mathbf{U}^{\mathsf{T}}.
    \]

4. Rotate and translate the centred frame back onto the reference centroid,
   \(\mathbf{P}_{\mathrm{aligned}} = \tilde{\mathbf{P}}\,\mathbf{R}_{\mathrm{rot}}^{\mathsf{T}} + \bar{\mathbf{R}}\).

This minimises the RMSD between each frame and the reference. The
\(\det\)-based correction guarantees a proper rotation (no mirror image).
Alignment can be disabled with `align=False` (or `--no-align`) when the
trajectory is already superimposed.

## PCA via SVD

After alignment, stack the frames into a data matrix
\(\mathbf{X} \in \mathbb{R}^{F\times D}\) (one flattened conformation per row)
and subtract the mean structure \(\boldsymbol{\mu}\):

\[
\mathbf{X}_c = \mathbf{X} - \boldsymbol{\mu}, \qquad
\boldsymbol{\mu} = \frac{1}{F}\sum_{f=1}^{F} \mathbf{X}_{f}.
\]

The classical approach diagonalises the \(D\times D\) covariance matrix
\(\mathbf{C} = \tfrac{1}{F-1}\,\mathbf{X}_c^{\mathsf{T}}\mathbf{X}_c\). QuickPCA
instead takes the **singular value decomposition** of the centred data directly,

\[
\mathbf{X}_c = \mathbf{U}\,\mathbf{S}\,\mathbf{V}^{\mathsf{T}},
\]

which avoids ever forming \(\mathbf{C}\). The two are equivalent: the right
singular vectors \(\mathbf{V}\) are the principal components (eigenvectors of
\(\mathbf{C}\)), and the eigenvalues — the variance explained by each mode — are

\[
\lambda_k = \frac{s_k^2}{F-1},
\]

where \(s_k\) is the \(k\)-th singular value. The **explained-variance ratio** is
\(\lambda_k / \sum_j \lambda_j\), and its running sum gives the cumulative
variance reported in the figure.

The **projections** (principal-component scores) are the coordinates of each
frame in the essential subspace,

\[
\mathbf{Z} = \mathbf{X}_c\,\mathbf{V}_{[:k]}^{\mathsf{T}},
\]

so column \(j\) of \(\mathbf{Z}\) is the time series of \(\mathrm{PC}_{j+1}\). The
columns of \(\mathbf{U}\) and \(\mathbf{V}\) are sign-flipped with the same
convention as scikit-learn (`svd_flip`), making the components deterministic and
identical to `PCA(svd_solver="full")`.

!!! note "Why SVD?"
    Diagonalising the \(3N\times 3N\) covariance matrix scales poorly with atom
    count and amplifies round-off error. The SVD of the \((F\times 3N)\) matrix
    delivers the same modes faster and more stably — the key efficiency trick
    behind QuickPCA.

## Residue cross-correlation

QuickPCA recovers the dynamic cross-correlation map **analytically from the PCA
modes**, without revisiting the raw trajectory. Reshape each eigenvector into
per-atom \(3\)-vectors, \(\mathbf{v}_k \to \mathbf{v}_{k,i}\in\mathbb{R}^3\), and
reconstruct the displacement covariance weighted by the eigenvalues:

\[
\mathbf{C}_{ij} =
\sum_{k} |\lambda_k|\;\mathbf{v}_{k,i}\cdot\mathbf{v}_{k,j}.
\]

Normalising to correlation form gives the familiar map in \([-1, 1]\):

\[
C^{\mathrm{norm}}_{ij} =
\frac{\mathbf{C}_{ij}}{\sqrt{\mathbf{C}_{ii}\,\mathbf{C}_{jj}}}.
\]

Values near \(+1\) indicate atoms (residues) that move together; values near
\(-1\) indicate anticorrelated motion. The diagonal is \(\approx 1\) by
construction. Because only the retained modes contribute, the map reflects the
**essential**, large-amplitude correlations rather than fast local noise.

## Free-Energy Landscape

Projecting the trajectory onto the first two components, \((\mathrm{PC}_1,
\mathrm{PC}_2)\), and estimating their probability density \(\rho\) lets us turn
sampling into a **free-energy surface** by **Boltzmann inversion**:

\[
F(\mathrm{PC}_1, \mathrm{PC}_2) = -k_B T \,\ln \rho(\mathrm{PC}_1, \mathrm{PC}_2).
\]

In QuickPCA the density \(\rho\) is a 2-D histogram of the projections
(`n_bins` bins per axis, with a 20 % padding around the data range), smoothed by
a Gaussian filter of width `sigma`. The thermal energy uses the gas constant in
kJ mol⁻¹ K⁻¹,

\[
k_B T = 0.008314462 \times T \ \ [\mathrm{kJ\,mol^{-1}}],
\]

so at \(T = 300\,\mathrm{K}\), \(k_B T \approx 2.494\ \mathrm{kJ\,mol^{-1}}\). The
surface is shifted so its global minimum sits at \(F = 0\); empty bins
(\(\rho = 0\)) map to undefined free energy and are masked out.

Deep basins in \(F\) correspond to densely sampled, low-energy conformational
states; ridges between them are transition regions. The report overlays the
trajectory path with **start** and **end** markers so you can read the
conformational journey directly off the landscape.

---

## References

- A. Amadei, A. B. M. Linssen, H. J. C. Berendsen.
  *Essential dynamics of proteins.* **Proteins** 17 (1993) 412–425.
- W. Kabsch. *A solution for the best rotation to relate two sets of vectors.*
  **Acta Crystallogr. A** 32 (1976) 922–923.
- C. C. David, D. J. Jacobs.
  *Principal Component Analysis: A Method for Determining the Essential Dynamics
  of Proteins.* **Methods Mol. Biol.** 1084 (2014) 193–226.
