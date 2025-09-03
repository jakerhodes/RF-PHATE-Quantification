# The files are too large to store on GitHub. They can be reproduced using splatter-data.R
# --- your existing setup & simulation (with no dropout in the simulator) ---
# Keep your installs; then:
library(splatter)
library(Matrix)
library(rlang)

# Example: 3 cell types (adjust as you like). No simulator dropout.
params <- newSplatParams(
  nGenes = 10000,
  batchCells = 5000,
  group.prob = c(0.3, 0.3, 0.4),
  dropout.type = "none"   # <- important: simulate without dropout
)


# shrink DE signal + add modest heterogeneity
# Shrink DE signal + add modest heterogeneity
# setParams() arguments and their defaults:
#   de.prob     = 0.1    # Default: 0.1 (probability a gene is DE)
#   de.facLoc   = 0.1    # Default: 0.1 (meanlog of DE fold-change)
#   de.facScale = 0.4    # Default: 0.4 (sdlog of DE fold-change)
#   bcv.common  = 0.1    # Default: 0.1 (common BCV)
#   bcv.df      = 60     # Default: 60 (degrees of freedom for BCV)
#   lib.scale   = 0.2    # Default: 0.2 (library size log-normal sd)
#   out.prob    = 0.05   # Default: 0.05 (probability of outlier burst)

params <- setParams(params,
  de.prob     = 0.10,   # fewer DE genes (default: 0.1)
  de.facLoc   = 0.10,   # weaker DE fold-changes (meanlog, default: 0.1)
  de.facScale = 0.40,   # tighter FC spread (default: 0.4)
  bcv.common  = 0.10,   # more within-group variability (default: 0.1)
  bcv.df      = 60,     # default: 60
  lib.scale   = 0.20,   # narrower library-size spread (default: 0.2)
  out.prob    = 0.05    # no outlier bursts (default: 0.05)
)




set.seed(42)
sim <- splatSimulate(params, method = "paths", verbose = FALSE)
counts <- counts(sim)
counts_sparse <- Matrix(counts, sparse = TRUE)

# Utility to find script dir (as you had)
get_script_dir <- function() {
  if (!is.null(sys.frame(1)$ofile)) {
    dirname(normalizePath(sys.frame(1)$ofile))
  } else {
    getwd()
  }
}
script_dir <- get_script_dir()

# Save labels once (shared for all dropout variants)
write.table(rownames(counts), file.path(script_dir, "synthetic_genes.txt"),
            quote = FALSE, row.names = FALSE, col.names = FALSE)
write.table(colnames(counts), file.path(script_dir, "synthetic_cells.txt"),
            quote = FALSE, row.names = FALSE, col.names = FALSE)
write.table(sim$Group, file.path(script_dir, "synthetic_cell_types.txt"),
            quote = FALSE, row.names = FALSE, col.names = FALSE)

# --- Post hoc dropout that preserves the baseline counts otherwise ---

# Apply dropout by setting a proportion of NON-ZERO entries to zero.
# By default this is done PER CELL so each cell loses ~rate of its nonzeros.
apply_dropout <- function(mat, rate, seed = NULL, per_cell = TRUE) {
  if (rate <= 0) return(mat)
  if (!inherits(mat, "dgCMatrix")) mat <- as(mat, "dgCMatrix")
  if (!is.null(seed)) set.seed(seed)

  p <- mat@p       # 0-based column pointers (length ncol+1)
  x <- mat@x       # nonzero values

  if (per_cell) {
    nc <- ncol(mat)
    for (j in seq_len(nc)) {
      start <- p[j] + 1L
      end   <- p[j + 1L]
      if (end >= start) {
        nnz_j <- end - start + 1L
        ndrop <- floor(nnz_j * rate)
        if (ndrop > 0L) {
          idx <- seq.int(start, end)
          drop_idx <- sample(idx, ndrop, replace = FALSE)
          x[drop_idx] <- 0
        }
      }
    }
  } else {
    # Global dropout across the whole matrix (not per cell)
    nnz <- length(x)
    ndrop <- floor(nnz * rate)
    if (ndrop > 0L) {
      drop_idx <- sample.int(nnz, ndrop, replace = FALSE)
      x[drop_idx] <- 0
    }
  }

  mat@x <- x
  Matrix::drop0(mat)  # remove structural zeros efficiently
}

# Generate and save versions at 0%,10%,...,50% dropout
rates <- seq(0, 0.5, by = 0.1)
for (r in rates) {
  # Seed per rate for reproducibility while keeping baseline identical
  mat_r <- if (r == 0) counts_sparse else apply_dropout(counts_sparse, r, seed = 202500 + as.integer(r * 1000))
  fname <- sprintf("synthetic_counts_dropout_%02dp.mtx", as.integer(r * 100))
  writeMM(mat_r, file.path(script_dir, fname))
}

# For convenience, also keep a copy matching your original name (no dropout)
writeMM(counts_sparse, file.path(script_dir, "synthetic_counts.mtx"))
