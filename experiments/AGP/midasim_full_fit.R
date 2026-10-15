#!/usr/bin/env Rscript
# Full-template MIDASim generation for the audited AGP selection/final cohorts.
# The established relative-only helpers are loaded from the verified snapshot.

agp_full_require <- function(ok, message) {
  if (!isTRUE(ok)) stop(message, call. = FALSE)
}

agp_full_main <- function(args = commandArgs(trailingOnly = TRUE)) {
  agp_full_require(length(args) == 2L && identical(args[[1L]], "--output"),
                   "Usage: midasim_full_fit.R --output GENERATOR_ROOT")
  for (name in c("jsonlite", "digest")) {
    agp_full_require(requireNamespace(name, quietly = TRUE), paste("Missing R package:", name))
  }
  output <- normalizePath(args[[2L]], mustWork = TRUE)
  sha <- function(path) {
    agp_full_require(file.exists(path), paste("Missing receipt input:", path))
    digest::digest(file = path, algo = "sha256", serialize = FALSE)
  }
  protocol_path <- file.path(output, "protocol.json")
  protocol <- jsonlite::read_json(protocol_path, simplifyVector = FALSE)
  agp_full_require(!is.null(protocol$config_sha256) &&
                     identical(sha(file.path(output, "config.json")), protocol$config_sha256),
                   "Full-generator configuration changed after preparation")
  source_names <- c("experiments/AGP/midasim_full_fit.R", "experiments/AGP/midasim_fit.R")
  agp_full_require(all(source_names %in% names(protocol$source_sha256)),
                   "Both full-generator and unchanged helper source receipts are required")
  script_flag <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  agp_full_require(length(script_flag) == 1L, "Cannot identify the executing full-generator script")
  script <- normalizePath(sub("^--file=", "", script_flag), mustWork = TRUE)
  agp_full_require(identical(sha(script), protocol$source_sha256[[source_names[[1L]]]]),
                   "Executing full-generator source differs from its frozen receipt")
  for (name in names(protocol$source_sha256)) {
    agp_full_require(!grepl("(^/|(^|/)\\.\\.(/|$))", name), "Invalid source receipt path")
    agp_full_require(identical(sha(file.path(output, "source", name)), protocol$source_sha256[[name]]),
                     paste("Frozen full-generator source changed:", name))
  }
  required_data <- c("data/generator_train.csv", "vendor/MIDASim_2.0.tar.gz")
  agp_full_require(all(required_data %in% names(protocol$data_receipts)),
                   "The generator template and vendor tarball require parent data receipts")
  for (name in names(protocol$data_receipts)) {
    agp_full_require(!grepl("(^/|(^|/)\\.\\.(/|$))", name), "Invalid data receipt path")
    agp_full_require(identical(sha(file.path(output, name)), protocol$data_receipts[[name]]),
                     paste("Frozen full-generator data changed:", name))
  }
  helper <- file.path(output, "source", source_names[[2L]])
  source(helper, local = TRUE)
  agp_dependencies()
  config <- jsonlite::read_json(file.path(output, "config.json"), simplifyVector = FALSE)
  agp_require(identical(config$mode, "nonparametric"), "Full generation requires nonparametric mode")
  checked_integer <- function(value, name) {
    agp_require(length(value) == 1L && is.numeric(value) && is.finite(value) &&
                  value >= 1 && value == floor(value) && value <= .Machine$integer.max,
                paste("Invalid frozen integer setting:", name))
    as.integer(value)
  }
  seed <- checked_integer(config$generation_seed, "generation_seed")
  tie_repetitions <- checked_integer(config$n_break_ties, "n_break_ties")
  role_names <- c("selection_train", "selection_stopping", "selection_calibration",
                  "selection_evaluation", "final_train", "final_stopping", "final_test")
  agp_require(identical(unlist(config$role_order, use.names = FALSE), role_names),
              "Frozen role_order must preserve the specified seven-role seed schedule")
  agp_require(setequal(names(config$roles), role_names) && length(config$roles) == length(role_names),
              "Frozen roles must specify the four selection and three final roles")
  role_sizes <- setNames(vapply(role_names, function(name) checked_integer(config$roles[[name]], name),
                                integer(1L)), role_names)
  generation_seed_offset <- 100000L
  agp_require(as.double(seed) + generation_seed_offset + 1000 * (length(role_names) - 1L) + 1 < .Machine$integer.max,
              "The full-generator seed schedule would overflow R integers")
  input <- file.path(output, "data", "generator_train.csv")
  vendor <- file.path(output, "vendor", "MIDASim_2.0.tar.gz")
  generation <- file.path(output, "generation")
  agp_require(!dir.exists(generation) || length(list.files(generation, all.files = TRUE, no.. = TRUE)) == 0L,
              "Generation requires a fresh generation directory; existing outputs are immutable")
  values <- as.matrix(utils::read.csv(input, header = FALSE, check.names = FALSE))
  template <- agp_template(values)
  agp_require(max(role_sizes) <= nrow(values), "Each role must be a subset of its own full-template-size draw")
  if (!is.null(config$generator_template_n)) {
    agp_require(nrow(values) == checked_integer(config$generator_template_n, "generator_template_n"),
                "Template row count differs from the frozen configuration")
  }
  dir.create(generation, recursive = TRUE, showWarnings = FALSE)
  base::options(digits = 17)
  if (!nzchar(Sys.getenv("TZ"))) Sys.setenv(TZ = "UTC")
  RNGkind("Mersenne-Twister", "Inversion", "Rejection")
  run_start <- proc.time()
  error_state <- list(stage = "setup", role = NULL)
  tryCatch({
    taxon_map <- data.frame(taxon_index = seq_len(ncol(values)) - 1L,
                            taxon_name = template$original_names,
                            training_positive_count = as.integer(template$prevalence),
                            included_in_fit = seq_len(ncol(values)) %in% template$active)
    utils::write.csv(taxon_map, file.path(generation, "taxon_map.csv"), row.names = FALSE, quote = TRUE)
    agp_json(file.path(generation, "input_metadata.json"), list(
      input = input, input_sha256 = sha(input), rows = nrow(values), taxa = ncol(values),
      active_taxa_indices = as.list(unname(template$active - 1L)),
      restored_zero_taxa_indices = as.list(unname(template$excluded - 1L)),
      indices_zero_based = TRUE, source_sha256 = sha(script), helper_sha256 = sha(helper),
      config_sha256 = sha(file.path(output, "config.json")), protocol_sha256 = sha(protocol_path),
      template_row_sums_are_not_read_depths = TRUE,
      template_row_sums_range = range(template$template_row_sums),
      setup_seed = seed, generation_seed_offset = generation_seed_offset,
      role_order = role_names, role_sizes = as.list(role_sizes)))
    set.seed(seed)
    setup <- agp_timed(function() MIDASim::MIDASim.setup(
      template$relative, n.break.ties = tie_repetitions, mode = "nonparametric"), "setup")
    fitted <- agp_strip_depth(setup$value)
    saveRDS(fitted, file.path(generation, "fitted.rds"), version = 3)
    message("Setup completed in ", format(setup$timing$elapsed_seconds, digits = 6), " s")
    error_state$stage <- "modify"
    modification <- agp_timed(function() MIDASim::MIDASim.modify(fitted), "modify")
    modified <- modification$value
    agp_require(is.null(modified$lib.size), "modify() unexpectedly inserted library depths")
    agp_require(all(is.finite(modified$theta)) && all(is.finite(modified$eta)),
                "MIDASim presence-absence fitting returned nonfinite parameters")
    saveRDS(modified, file.path(generation, "modified.rds"), version = 3)
    message("Modification completed in ", format(modification$timing$elapsed_seconds, digits = 6), " s")
    roles <- list()
    for (index in seq_along(role_names)) {
      role <- role_names[[index]]
      generation_seed <- as.integer(seed + generation_seed_offset + 1000 * (index - 1L))
      subset_seed <- generation_seed + 1L
      error_state <- list(stage = "sample", role = role)
      draw <- agp_timed(function() agp_draw(modified, template, generation_seed), role)
      set.seed(subset_seed)
      selected <- sample.int(nrow(draw$value), size = role_sizes[[role]], replace = FALSE)
      subset <- draw$value[selected, , drop = FALSE]
      agp_check_relative(subset)
      file <- file.path(generation, paste0(role, ".csv"))
      utils::write.table(subset, file, sep = ",", row.names = FALSE, col.names = FALSE, quote = FALSE)
      restored <- as.matrix(utils::read.csv(file, header = FALSE, check.names = FALSE))
      agp_check_relative(restored)
      agp_require(max(abs(restored - subset)) <= 1e-14, "Relative-abundance CSV roundtrip changed values")
      indices_file <- file.path(generation, paste0(role, "_indices.csv"))
      utils::write.csv(data.frame(output_row = seq_along(selected) - 1L, generated_row = selected - 1L),
                       indices_file, row.names = FALSE, quote = FALSE)
      detail <- list(role = role, role_index = index - 1L, rows = nrow(subset), taxa = ncol(subset),
        generation_seed = generation_seed, subset_seed = subset_seed,
        generated_rows_before_subsampling = nrow(draw$value), indices_zero_based = TRUE,
        path = paste0("generation/", role, ".csv"), sha256 = sha(file), indices_sha256 = sha(indices_file),
        timing = draw$timing, warnings = draw$warnings,
        mean_positive_taxa = mean(rowSums(subset > 0)), positive_taxa_range = range(rowSums(subset > 0)),
        max_row_sum_error = max(abs(rowSums(subset) - 1)))
      agp_json(file.path(generation, paste0(role, "_metrics.json")), detail)
      roles[[role]] <- detail
      message(role, ": ", nrow(subset), " x ", ncol(subset), "; generation ",
              format(draw$timing$elapsed_seconds, digits = 6), " s")
    }
    # Capture final namespaces, including lazily loaded MASS/pracma/psych.
    namespaces <- sort(loadedNamespaces())
    package_versions <- setNames(lapply(namespaces, function(name) as.character(utils::packageVersion(name))), namespaces)
    agp_json(file.path(generation, "package_versions.json"), package_versions)
    utils::capture.output(utils::sessionInfo(), file = file.path(generation, "sessionInfo.txt"))
    installed <- system.file(package = "MIDASim")
    installed_files <- list.files(installed, recursive = TRUE, full.names = TRUE)
    installed_hashes <- setNames(lapply(installed_files, sha), substring(installed_files, nchar(installed) + 2L))
    total <- proc.time() - run_start
    metrics <- list(status = "complete", simulator = "MIDASim", version = "2.0",
      mode = "nonparametric", only_relative_abundance = TRUE, count_data_generated = FALSE,
      fabricated_library_depths = FALSE, library_size_metadata_removed = TRUE,
      input = list(path = input, sha256 = sha(input), rows = nrow(values), taxa = ncol(values)),
      source_sha256 = sha(script), helper_sha256 = sha(helper),
      config_sha256 = sha(file.path(output, "config.json")), protocol_sha256 = sha(protocol_path),
      package_source = list(path = vendor, sha256 = sha(vendor)),
      installed_package = list(path = installed, sha256 = installed_hashes), package_versions = package_versions,
      seed = seed, setup_seed = seed, generation_seed_offset = generation_seed_offset,
      rng_kind = RNGkind(), tie_break_repetitions = tie_repetitions,
      fitted_taxa = length(template$active), restored_zero_taxa_indices = as.list(unname(template$excluded - 1L)),
      original_dimension = template$original_dimension,
      train_only_zero_policy = "Remove taxa unobserved in the template before fitting; restore exact-zero columns in original feature order",
      setup = c(setup$timing, list(warnings = setup$warnings)),
      modify = c(modification$timing, list(warnings = modification$warnings)),
      total_elapsed_seconds = unname(total[["elapsed"]]), role_order = role_names, roles = roles,
      draw_protocol = "Each of seven roles is an independent full-template-size MIDASim call followed by a separately seeded uniform subset/permutation",
      limitations = c("MIDASim rank matching couples observations within a generated table; diagnostics are descriptive, not IID coverage evidence",
                      "No read counts or sequencing-depth response are modeled in this relative-abundance-only experiment",
                      "Training-unobserved taxa remain absent from generated samples"))
    agp_json(file.path(generation, "metrics.json"), metrics)
    files <- list.files(generation, recursive = TRUE, full.names = TRUE)
    hashes <- setNames(lapply(files, sha), substring(files, nchar(output) + 2L))
    agp_json(file.path(generation, "COMPLETE.json"), list(status = "complete",
      finished_utc = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
      input_sha256 = sha(input), source_sha256 = sha(script), helper_sha256 = sha(helper),
      config_sha256 = metrics$config_sha256, protocol_sha256 = metrics$protocol_sha256,
      role_order = role_names, outputs = hashes))
    invisible(metrics)
  }, error = function(condition) {
    agp_json(file.path(generation, "FAILED.json"), c(error_state, list(message = conditionMessage(condition))))
    stop(condition)
  })
}

if (sys.nframe() == 0L) agp_full_main()
