library(splatter)
library(Matrix)
library(rlang)

# Minimal noise: reduce variability, outliers, and DE signal
params <- newSplatParams(
    nGenes = 10000,
    batchCells = 5000,
    group.prob = c(0.3, 0.3, 0.4),
    dropout.type = "none"
)

params <- setParams(params,
    de.prob     = 0.001,   # Even fewer DE genes
    de.facLoc   = 0.001,   # Even weaker DE fold-changes
    de.facScale = 0.001,   # Minimal FC spread
    bcv.common  = 0.001,   # Even less within-group variability
    bcv.df      = 10000,   # Higher df for more stable BCV
    lib.scale   = 0.001,   # Even less library-size spread
    out.prob    = 0.0      # No outlier bursts
)

set.seed(42)
sim <- splatSimulate(params, method = "paths", verbose = FALSE)
counts <- counts(sim)
counts_sparse <- Matrix(counts, sparse = TRUE)

get_script_dir <- function() {
    if (!is.null(sys.frame(1)$ofile)) {
        dirname(normalizePath(sys.frame(1)$ofile))
    } else {
        getwd()
    }
}
script_dir <- get_script_dir()

write.table(rownames(counts), file.path(script_dir, "synthetic_genes_paths.txt"),
                        quote = FALSE, row.names = FALSE, col.names = FALSE)
write.table(colnames(counts), file.path(script_dir, "synthetic_cells_paths.txt"),
                        quote = FALSE, row.names = FALSE, col.names = FALSE)
write.table(sim$Group, file.path(script_dir, "synthetic_cell_types_paths.txt"),
                        quote = FALSE, row.names = FALSE, col.names = FALSE)

apply_dropout <- function(mat, rate, seed = NULL, per_cell = TRUE) {
    if (rate <= 0) return(mat)
    if (!inherits(mat, "dgCMatrix")) mat <- as(mat, "dgCMatrix")
    if (!is.null(seed)) set.seed(seed)
    p <- mat@p
    x <- mat@x
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
        nnz <- length(x)
        ndrop <- floor(nnz * rate)
        if (ndrop > 0L) {
            drop_idx <- sample.int(nnz, ndrop, replace = FALSE)
            x[drop_idx] <- 0
        }
    }
    mat@x <- x
    Matrix::drop0(mat)
}

rates <- seq(0, 0.5, by = 0.1)
for (r in rates) {
    mat_r <- if (r == 0) counts_sparse else apply_dropout(counts_sparse, r, seed = 202500 + as.integer(r * 1000))
    fname <- sprintf("synthetic_counts_dropout_%02dp_paths.mtx", as.integer(r * 100))
    writeMM(mat_r, file.path(script_dir, fname))
}

writeMM(counts_sparse, file.path(script_dir, "synthetic_counts_paths.mtx"))
