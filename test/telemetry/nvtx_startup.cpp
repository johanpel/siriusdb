// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0

#include "config.hpp"
#include "duckdb.hpp"
#include "log/logging.hpp"
#include "sirius/ffi.hpp"
#include "sirius_extension.hpp"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string_view>

// This helper intentionally has no test fixtures: the first extension load must
// exercise process startup before NVTX or Quent have cached any state.
int main(int argc, char** argv)
{
  const std::string_view mode = argc == 2 ? argv[1] : "duckdb";
  const bool logging          = mode.find("log") != std::string_view::npos;
  const auto* rejection       = std::getenv("SIRIUS_TEST_QUENT_REJECT");
  const bool expect_rejection = rejection && std::string_view(rejection) == "1";
  const auto previous_sink    = sirius::log::get_sink();
  const auto previous_backend = duckdb::Config::LOG_BACKEND;
  const auto previous_dir     = duckdb::Config::LOG_DIR;
  const auto previous_level   = duckdb::Config::LOG_LEVEL;
  try {
    if (mode == "ffi" || mode == "ffi-log") {
      auto const* path = std::getenv("SIRIUS_CONFIG_FILE");
      if (path == nullptr) { throw std::runtime_error("missing FFI config path"); }
      sirius::ffi::Context context{path};
      if (logging) {
        sirius::log::get_sink()->log(
          sirius::log::level::critical, std::source_location::current(), "quent startup marker");
      }
      return expect_rejection ? 1 : 0;
    }
    duckdb::DuckDB db(nullptr);
    db.LoadStaticExtension<duckdb::SiriusExtension>();
    duckdb::Connection connection(db);
    if (logging) {
      if (mode == "duckdb-log-sql") {
        auto setting = connection.Query("SET sirius_log_backend = 'quent'");
        if (setting->HasError()) { throw std::runtime_error(setting->GetError()); }
      }
      auto level = connection.Query("SET sirius_log_level = 'off'");
      if (level->HasError()) { throw std::runtime_error(level->GetError()); }
      sirius::log::get_sink()->log(
        sirius::log::level::critical, std::source_location::current(), "filtered startup marker");
      level = connection.Query("SET sirius_log_level = 'critical'");
      if (level->HasError()) { throw std::runtime_error(level->GetError()); }
      auto setting = connection.Query("SET sirius_log_backend = 'noop'");
      if (setting->HasError()) { throw std::runtime_error(setting->GetError()); }
      setting = connection.Query("SET sirius_log_backend = 'quent'");
      if (setting->HasError()) { throw std::runtime_error(setting->GetError()); }
      sirius::log::get_sink()->log(
        sirius::log::level::critical, std::source_location::current(), "quent startup marker");
      if (expect_rejection) { return 1; }
    }
    auto settings = connection.Query("SET enable_duckdb_fallback = false");
    if (settings->HasError()) { throw std::runtime_error(settings->GetError()); }
    auto result = connection.Query(
      "SELECT k, sum(v)::BIGINT FROM (VALUES (1, 10), (1, 20), (2, 5)) t(k, v) "
      "GROUP BY k ORDER BY k");
    if (result->HasError()) { throw std::runtime_error(result->GetError()); }
    if (result->RowCount() != 2 || result->GetValue(0, 0).GetValue<int32_t>() != 1 ||
        result->GetValue(1, 0).GetValue<int64_t>() != 30 ||
        result->GetValue(0, 1).GetValue<int32_t>() != 2 ||
        result->GetValue(1, 1).GetValue<int64_t>() != 5) {
      throw std::runtime_error("unexpected GPU result");
    }
  } catch (const std::exception& error) {
    const bool logger_restored =
      mode == "duckdb-log-sql"
        ? duckdb::Config::LOG_BACKEND == "noop" &&
            !sirius::log::get_sink()->should_log(sirius::log::level::critical)
        : sirius::log::get_sink() == previous_sink &&
            duckdb::Config::LOG_BACKEND == previous_backend &&
            duckdb::Config::LOG_DIR == previous_dir && duckdb::Config::LOG_LEVEL == previous_level;
    if (expect_rejection && logger_restored &&
        std::string_view(error.what()).find("enabled engine telemetry session") !=
          std::string_view::npos) {
      return 0;
    }
    std::cerr << error.what() << '\n';
    return 1;
  }
}
