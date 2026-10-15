#!/usr/bin/env Rscript
# MIDASim 2.0 nonparametric, relative-abundance-only AGP pilot.
# No library depths are assumed: the unused lib.size field is removed before
# modify() and simulation. Each bank/role is a separate full-template-size draw.

agp_require <- function(ok, message) {
  if (!isTRUE(ok)) stop(message, call. = FALSE)
}

agp_sha256 <- function(path) {
  agp_require(file.exists(path), paste("Missing receipt input:", path))
  digest::digest(file = path, algo = "sha256", serialize = FALSE)
}

agp_json <- function(path, value) {
  jsonlite::write_json(value, path, auto_unbox = TRUE, pretty = TRUE,
                      digits = 17, null = "null", na = "null")
}

agp_dependencies <- function() {
  for (name in c("MIDASim", "jsonlite", "digest")) {
    agp_require(requireNamespace(name, quietly = TRUE), paste("Missing R package:", name))
  }
  agp_require(as.character(utils::packageVersion("MIDASim")) == "2.0",
              "This audited workflow requires MIDASim version 2.0")
}

agp_template <- function(values) {
  values <- as.matrix(values)
  storage.mode(values) <- "double"
  agp_require(nrow(values) >= 2L && ncol(values) >= 2L,
              "The template requires at least two observations and two taxa")
  agp_require(all(is.finite(values)) && all(values >= 0) && all(rowSums(values) > 0),
              "The template must be finite, nonnegative, and have no empty observations")
  prevalence <- colSums(values > 0)
  active <- which(prevalence > 0)
  agp_require(length(active) >= 2L, "At least two taxa must occur in the training template")
  # CRAN examples recommend at least two presences; fail rather than silently
  # remove rare-but-observed taxa. The designated AGP template has no singletons.
  agp_require(!any(prevalence == 1L), "A singleton training taxon needs an explicit sampling policy")
  colnames(values) <- sprintf("taxon_%04d", seq_len(ncol(values)) - 1L)
  relative <- values / rowSums(values)
  list(relative = relative[, active, drop = FALSE], active = active,
       excluded = which(prevalence == 0), original_dimension = ncol(values),
       original_names = colnames(values), prevalence = prevalence,
       template_row_sums = rowSums(values))
}

agp_strip_depth <- function(fitted) {
  agp_require(identical(fitted$mode, "nonparametric"), "Relative-only sampling requires nonparametric mode")
  fitted$lib.size <- NULL
  fitted
}

agp_check_relative <- function(values, presence = NULL, tolerance = 1e-10) {
  agp_require(is.matrix(values) && all(is.finite(values)) && all(values >= 0),
              "MIDASim returned invalid relative abundances")
  totals <- rowSums(values)
  agp_require(all(totals > 0), "MIDASim returned an empty sample; no sample was silently removed")
  agp_require(max(abs(totals - 1)) <= tolerance, "MIDASim relative-abundance rows do not sum to one")
  if (!is.null(presence)) {
    agp_require(identical(dim(values), dim(presence)) && all((values > 0) == (presence > 0)),
                "MIDASim presence masks and abundance zeros disagree")
  }
  invisible(TRUE)
}

agp_restore <- function(values, template) {
  agp_require(ncol(values) == length(template$active), "Generated active-taxon dimension changed")
  full <- matrix(0, nrow(values), template$original_dimension,
                 dimnames = list(NULL, template$original_names))
  full[, template$active] <- values
  agp_check_relative(full)
  full
}

agp_draw <- function(modified, template, seed) {
  agp_require(is.null(modified$lib.size), "Unused library-depth metadata must be stripped before generation")
  agp_require(identical(modified$mode, "nonparametric"), "A parametric MIDASim fit is outside this pilot")
  set.seed(as.integer(seed))
  result <- MIDASim::MIDASim(modified, only.rel = TRUE)
  agp_require(is.null(result$sim_count), "Relative-only MIDASim unexpectedly generated counts")
  agp_check_relative(result$sim_rel, result$sim_01)
  agp_require(nrow(result$sim_rel) == modified$n.sample,
              "MIDASim changed its requested full-template sample count")
  agp_restore(result$sim_rel, template)
}

agp_timed <- function(fun, phase) {
  warnings <- character()
  start <- proc.time()
  value <- withCallingHandlers(fun(), warning = function(condition) {
    warnings <<- c(warnings, conditionMessage(condition))
    invokeRestart("muffleWarning")
  })
  timing <- proc.time() - start
  list(value = value, timing = list(phase = phase,
       elapsed_seconds = unname(timing[["elapsed"]]),
       user_seconds = unname(timing[["user.self"]]),
       system_seconds = unname(timing[["sys.self"]])), warnings = unique(warnings))
}

agp_verify_protocol <- function(output, input, script) {
  path <- file.path(output, "protocol.json")
  agp_require(file.exists(path), "Parent protocol.json is required before generation")
  protocol <- jsonlite::read_json(path, simplifyVector = FALSE)
  agp_require(!is.null(protocol$config_sha256), "Parent protocol lacks the configuration receipt")
  agp_require(identical(agp_sha256(file.path(output, "config.json")), protocol$config_sha256),
              "Parent configuration changed after preparation")
  expected <- protocol$source_sha256[["experiments/AGP/midasim_fit.R"]]
  agp_require(!is.null(expected) && identical(agp_sha256(script), expected),
              "Executing MIDASim script differs from the parent source receipt")
  for (name in names(protocol$source_sha256)) {
    agp_require(!grepl("(^/|(^|/)\\.\\.(/|$))", name), "Invalid source receipt path")
    agp_require(identical(agp_sha256(file.path(output, "source", name)), protocol$source_sha256[[name]]),
                paste("Frozen source changed:", name))
  }
  expected_input <- normalizePath(file.path(output, "data", "generator_train.csv"), mustWork = TRUE)
  agp_require(identical(normalizePath(input, mustWork = TRUE), expected_input),
              "The input must be the parent's frozen data/generator_train.csv")
  agp_require(!is.null(protocol$data_receipts[["data/generator_train.csv"]]),
              "Parent protocol lacks the generator template receipt")
  for (name in names(protocol$data_receipts)) {
    agp_require(!grepl("(^/|(^|/)\\.\\.(/|$))", name), "Invalid data receipt path")
    agp_require(identical(agp_sha256(file.path(output, name)), protocol$data_receipts[[name]]),
                paste("Frozen parent data changed:", name))
  }
  invisible(protocol)
}

agp_arguments <- function(args) {
  options <- list(input = NULL, output = NULL, banks = 5L, seed = 2026100410L,
                  n_train = 4581L, n_stopping = 1044L, n_test = 2089L,
                  n_break_ties = 100L, package_source = NULL)
  agp_require(length(args) %% 2L == 0L, "Arguments require --name value pairs")
  if (length(args)) for (i in seq.int(1L, length(args), by = 2L)) {
    key <- gsub("-", "_", sub("^--", "", args[[i]]), fixed = TRUE)
    agp_require(startsWith(args[[i]], "--") && key %in% names(options), paste("Unknown argument:", args[[i]]))
    options[[key]] <- args[[i + 1L]]
  }
  agp_require(!is.null(options$input) && !is.null(options$output), "Both --input and --output are required")
  for (key in c("banks", "seed", "n_train", "n_stopping", "n_test", "n_break_ties")) {
    numeric <- suppressWarnings(as.numeric(options[[key]]))
    agp_require(length(numeric) == 1L && is.finite(numeric) && numeric >= 1 && numeric == floor(numeric)
                && numeric <= .Machine$integer.max, paste("Invalid integer option:", key))
    options[[key]] <- as.integer(numeric)
  }
  agp_require(options$seed + 100000 * options$banks + 3000 < .Machine$integer.max,
              "The per-bank seed schedule would overflow R's integer seed range")
  options
}

agp_main <- function(args = commandArgs(trailingOnly = TRUE)) {
  options <- agp_arguments(args)
  agp_dependencies()
  script_flag <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  agp_require(length(script_flag) == 1L, "Cannot identify the executing script")
  script <- normalizePath(sub("^--file=", "", script_flag), mustWork = TRUE)
  output <- normalizePath(options$output, mustWork = TRUE)
  input <- normalizePath(options$input, mustWork = TRUE)
  agp_verify_protocol(output, input, script)
  if (is.null(options$package_source) && file.exists(file.path(output, "vendor", "MIDASim_2.0.tar.gz"))) {
    options$package_source <- file.path(output, "vendor", "MIDASim_2.0.tar.gz")
  }
  config <- jsonlite::read_json(file.path(output, "config.json"), simplifyVector = TRUE)
  # The parent has a separate discriminator-training seed; only its
  # generation_seed belongs to this simulator's RNG schedule.
  expected_options <- list(banks = config$banks, seed = config$generation_seed,
                            n_train = config$n_q$train, n_stopping = config$n_q$stopping,
                            n_test = config$n_q$test, n_break_ties = config$n_break_ties)
  for (key in names(expected_options)) {
    agp_require(!is.null(expected_options[[key]]) && identical(as.integer(expected_options[[key]]), options[[key]]),
                paste("Command-line option differs from the frozen configuration:", key))
  }
  generation <- file.path(output, "generation")
  agp_require(!dir.exists(generation) || length(list.files(generation, all.files = TRUE, no.. = TRUE)) == 0L,
              "Generation requires a fresh generation directory; existing results are immutable")
  values <- as.matrix(utils::read.csv(input, header = FALSE, check.names = FALSE))
  template <- agp_template(values)
  agp_require(max(options$n_train, options$n_stopping, options$n_test) <= nrow(values),
              "Each role must be a subset of its own full-template-size MIDASim draw")
  dir.create(generation, recursive = TRUE, showWarnings = FALSE)
  base::options(digits = 17)
  if (!nzchar(Sys.getenv("TZ"))) Sys.setenv(TZ = "UTC")
  RNGkind("Mersenne-Twister", "Inversion", "Rejection")
  run_start <- proc.time()
  error_state <- list(stage = "setup", bank = NULL, role = NULL)
  tryCatch({
    namespaces <- sort(loadedNamespaces())
    package_versions <- setNames(lapply(namespaces, function(name) {
      as.character(utils::packageVersion(name))
    }), namespaces)
    agp_json(file.path(generation, "package_versions.json"), package_versions)
    taxon_map <- data.frame(taxon_index = seq_len(ncol(values)) - 1L,
                            taxon_name = template$original_names,
                            training_positive_count = as.integer(template$prevalence),
                            included_in_fit = seq_len(ncol(values)) %in% template$active)
    utils::write.csv(taxon_map, file.path(generation, "taxon_map.csv"), row.names = FALSE, quote = TRUE)
    agp_json(file.path(generation, "input_metadata.json"), list(
      input = input, input_sha256 = agp_sha256(input), rows = nrow(values),
      taxa = ncol(values), active_taxa_indices = template$active - 1L,
      restored_zero_taxa_indices = template$excluded - 1L,
      indices_zero_based = TRUE, source_sha256 = agp_sha256(script),
      config_sha256 = agp_sha256(file.path(output, "config.json")),
      protocol_sha256 = agp_sha256(file.path(output, "protocol.json")),
      template_row_sums_are_not_read_depths = TRUE,
      template_row_sums_range = range(template$template_row_sums)))
    set.seed(options$seed)
    setup <- agp_timed(function() MIDASim::MIDASim.setup(
      template$relative, n.break.ties = options$n_break_ties, mode = "nonparametric"), "setup")
    fitted <- agp_strip_depth(setup$value)
    saveRDS(fitted, file.path(generation, "fitted.rds"), version = 3)
    error_state$stage <- "modify"
    modification <- agp_timed(function() MIDASim::MIDASim.modify(fitted), "modify")
    modified <- modification$value
    agp_require(is.null(modified$lib.size), "modify() unexpectedly inserted library depths")
    agp_require(all(is.finite(modified$theta)) && all(is.finite(modified$eta)),
                "MIDASim presence-absence fitting returned nonfinite parameters")
    saveRDS(modified, file.path(generation, "modified.rds"), version = 3)
    banks <- list()
    role_sizes <- c(train = options$n_train, stopping = options$n_stopping, test = options$n_test)
    for (bank in seq_len(options$banks) - 1L) {
      bank_name <- sprintf("bank_%02d", bank)
      bank_directory <- file.path(generation, bank_name)
      dir.create(bank_directory)
      banks[[bank_name]] <- list()
      for (index in seq_along(role_sizes)) {
        role <- names(role_sizes)[[index]]
        generation_seed <- as.integer(options$seed + 100000 * (bank + 1L) + 1000 * (index - 1L))
        subset_seed <- generation_seed + 1L
        error_state <- list(stage = "sample", bank = bank, role = role)
        draw <- agp_timed(function() agp_draw(modified, template, generation_seed),
                          paste(bank_name, role, sep = "/"))
        set.seed(subset_seed)
        selected <- sample.int(nrow(draw$value), size = role_sizes[[index]], replace = FALSE)
        subset <- draw$value[selected, , drop = FALSE]
        agp_check_relative(subset)
        file <- file.path(bank_directory, paste0(role, ".csv"))
        utils::write.table(subset, file, sep = ",", row.names = FALSE, col.names = FALSE, quote = FALSE)
        # Verify the serialized values, not just the in-memory sample.
        restored <- as.matrix(utils::read.csv(file, header = FALSE, check.names = FALSE))
        agp_check_relative(restored)
        agp_require(max(abs(restored - subset)) <= 1e-14, "Relative-abundance CSV roundtrip changed values")
        indices_file <- file.path(bank_directory, paste0(role, "_indices.csv"))
        utils::write.csv(data.frame(output_row = seq_along(selected) - 1L,
                                   generated_row = selected - 1L), indices_file,
                         row.names = FALSE, quote = FALSE)
        detail <- list(bank = bank, role = role, rows = nrow(subset), taxa = ncol(subset),
                       generation_seed = generation_seed, subset_seed = subset_seed,
                       generated_rows_before_subsampling = nrow(draw$value),
                       indices_zero_based = TRUE,
                       path = paste("generation", bank_name, paste0(role, ".csv"), sep = "/"),
                       sha256 = agp_sha256(file), indices_sha256 = agp_sha256(indices_file),
                       timing = draw$timing, warnings = draw$warnings,
                       mean_positive_taxa = mean(rowSums(subset > 0)),
                       positive_taxa_range = range(rowSums(subset > 0)),
                       max_row_sum_error = max(abs(rowSums(subset) - 1)))
        agp_json(file.path(bank_directory, paste0(role, "_metrics.json")), detail)
        banks[[bank_name]][[role]] <- detail
        message(bank_name, "/", role, ": ", nrow(subset), " x ", ncol(subset),
                "; generation ", format(draw$timing$elapsed_seconds, digits = 5), " s")
      }
    }
    utils::capture.output(utils::sessionInfo(), file = file.path(generation, "sessionInfo.txt"))
    total <- proc.time() - run_start
    installed <- system.file(package = "MIDASim")
    installed_files <- list.files(installed, recursive = TRUE, full.names = TRUE)
    installed_hashes <- setNames(lapply(installed_files, agp_sha256),
                                substring(installed_files, nchar(installed) + 2L))
    metrics <- list(status = "complete", simulator = "MIDASim", version = "2.0",
      mode = "nonparametric", only_relative_abundance = TRUE, count_data_generated = FALSE,
      fabricated_library_depths = FALSE, library_size_metadata_removed = TRUE,
      input = list(path = input, sha256 = agp_sha256(input), rows = nrow(values), taxa = ncol(values)),
      source_sha256 = agp_sha256(script), config_sha256 = agp_sha256(file.path(output, "config.json")),
      protocol_sha256 = agp_sha256(file.path(output, "protocol.json")),
      package_source = if (is.null(options$package_source)) NULL else list(
        path = normalizePath(options$package_source, mustWork = TRUE), sha256 = agp_sha256(options$package_source)),
      installed_package = list(path = installed, sha256 = installed_hashes),
      package_versions = package_versions,
      seed = options$seed, rng_kind = RNGkind(), tie_break_repetitions = options$n_break_ties,
      fitted_taxa = length(template$active), restored_zero_taxa_indices = template$excluded - 1L,
      original_dimension = template$original_dimension,
      train_only_zero_policy = "Remove taxa unobserved in the template before fitting; restore exact-zero columns in original feature order",
      setup = c(setup$timing, list(warnings = setup$warnings)),
      modify = c(modification$timing, list(warnings = modification$warnings)),
      total_elapsed_seconds = unname(total[["elapsed"]]), banks = banks,
      draw_protocol = "Each bank/role is an independent full-template-size MIDASim call followed by a separately seeded uniform subset/permutation",
      limitations = c("MIDASim rank matching couples observations within a generated table; diagnostics are descriptive, not IID coverage evidence",
                      "No read counts or sequencing-depth response are modeled in this relative-abundance-only pilot",
                      "Training-unobserved taxa remain absent from generated samples"))
    agp_json(file.path(generation, "metrics.json"), metrics)
    files <- list.files(generation, recursive = TRUE, full.names = TRUE)
    hashes <- setNames(lapply(files, agp_sha256), substring(files, nchar(output) + 2L))
    agp_json(file.path(generation, "COMPLETE.json"), list(status = "complete",
      finished_utc = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
      input_sha256 = agp_sha256(input), source_sha256 = agp_sha256(script),
      config_sha256 = metrics$config_sha256, protocol_sha256 = metrics$protocol_sha256,
      banks = options$banks, outputs = hashes))
    invisible(metrics)
  }, error = function(condition) {
    agp_json(file.path(generation, "FAILED.json"), c(error_state, list(message = conditionMessage(condition))))
    stop(condition)
  })
}

if (sys.nframe() == 0L) agp_main()
