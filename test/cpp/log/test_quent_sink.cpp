/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "catch.hpp"
#include "log/logging.hpp"
#include "log/quent_sink.hpp"
#include "sirius_config.hpp"
#include "telemetry/telemetry_context.hpp"
#include "utils/log_test_utils.hpp"
#include "yyjson.hpp"

#include <unistd.h>

#include <array>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <latch>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
namespace json = duckdb_yyjson;
using sirius::log::level;

struct log_record {
  std::string id;
  std::string severity;
  std::string message;
  std::string file;
  uint32_t line;
  std::string function;
  std::string worker;
};

std::string string_field(json::yyjson_val* value, const char* key)
{
  auto* field = json::yyjson_obj_get(value, key);
  return json::yyjson_is_str(field)
           ? std::string(json::yyjson_get_str(field), json::yyjson_get_len(field))
           : std::string{};
}

std::vector<log_record> read_records(const std::filesystem::path& root)
{
  std::vector<log_record> records;
  for (const auto& entry : std::filesystem::recursive_directory_iterator(root)) {
    if (entry.path().parent_path().filename() != "Log" || entry.path().extension() != ".ndjson") {
      continue;
    }
    std::ifstream input(entry.path());
    std::string line;
    while (std::getline(input, line)) {
      auto doc = std::unique_ptr<json::yyjson_doc, decltype(&json::yyjson_doc_free)>(
        json::yyjson_read(line.data(), line.size(), 0), &json::yyjson_doc_free);
      REQUIRE(doc != nullptr);
      auto* event   = json::yyjson_doc_get_root(doc.get());
      auto* data    = json::yyjson_obj_get(event, "data");
      auto iterator = json::yyjson_obj_iter_with(data);
      auto* key     = json::yyjson_obj_iter_next(&iterator);
      REQUIRE(key != nullptr);
      auto* payload = json::yyjson_obj_iter_get_val(key);
      records.push_back(
        {string_field(event, "id"),
         json::yyjson_get_str(key),
         string_field(payload, "message"),
         string_field(payload, "file"),
         static_cast<uint32_t>(json::yyjson_get_uint(json::yyjson_obj_get(payload, "line"))),
         string_field(payload, "function"),
         string_field(json::yyjson_obj_get(payload, "worker_id"), "target")});
    }
  }
  return records;
}

struct telemetry_fixture {
  std::filesystem::path root = std::filesystem::temp_directory_path() /
                               ("sirius-quent-log-" + quent::to_string(quent::now_v7()));
  std::shared_ptr<const sirius::telemetry::telemetry_context> context;

  telemetry_fixture()
  {
    sirius::telemetry_config config;
    config.output_directory = root.string();
    context                 = sirius::telemetry::telemetry_context::create(
      sirius::telemetry::make_quent_context(config), config);
  }

  ~telemetry_fixture()
  {
    context.reset();
    std::error_code error;
    std::filesystem::remove_all(root, error);
  }
};
}  // namespace

TEST_CASE("Quent sink requires enabled telemetry", "[log][quent_sink]")
{
  REQUIRE_THROWS_AS(sirius::log::make_quent_sink({}), std::invalid_argument);
  sirius::telemetry_config config;
  config.enable_quent = false;
  auto context = sirius::telemetry::telemetry_context::create(quent::Context::none(), config);
  REQUIRE_THROWS_AS(sirius::log::make_quent_sink(context), std::invalid_argument);
  std::weak_ptr<const sirius::telemetry::telemetry_context> expired = context;
  context.reset();
  REQUIRE_THROWS_AS(sirius::log::make_quent_sink(expired), std::invalid_argument);
}

TEST_CASE("Quent sink preserves severity, text, source and session identity", "[log][quent_sink]")
{
  telemetry_fixture fixture;
  auto sink = sirius::log::make_quent_sink(fixture.context);
  CHECK_FALSE(sink->should_log(level::trace));
  CHECK(sink->should_log(level::info));
  CHECK_FALSE(sink->should_log(level::off));
  const auto worker = quent::to_string(fixture.context->worker_id().raw());
  const auto log_id = quent::to_string(fixture.context->log_handle().id().raw());
  const auto source = std::source_location::current();
  const std::string message{"quotes \" and newline\n and braces {}"};
  const std::array levels{
    level::trace, level::debug, level::info, level::warn, level::error, level::critical};
  const std::array names{"Trace", "Debug", "Info", "Warn", "Error", "Critical"};
  sink->set_level(level::trace);
  for (auto lvl : levels) {
    sink->log(lvl, source, message);
  }
  sink->log(level::off, source, "discarded");
  sink->set_level(level::warn);
  CHECK_FALSE(sink->should_log(level::info));
  CHECK(sink->should_log(level::warn));
  sink->log(level::info, source, "discarded");
  sink->log(level::warn, source, "threshold");
  sink->set_level(level::off);
  for (auto lvl : levels) {
    CHECK_FALSE(sink->should_log(lvl));
    sink->log(lvl, source, "discarded");
  }
  CHECK_FALSE(sink->flush());
  fixture.context.reset();
  sink->set_level(level::trace);
  CHECK_FALSE(sink->should_log(level::critical));
  sink->log(level::critical, source, "expired");
  const auto records = read_records(fixture.root);
  REQUIRE(records.size() == 8);
  CHECK(records.front().severity == "Initialized");
  CHECK(records.front().worker == worker);
  for (size_t index = 0; index < levels.size(); ++index) {
    const auto& record = records[index + 1];
    CHECK(record.id == log_id);
    CHECK(record.severity == names[index]);
    CHECK(record.message == message);
    CHECK(record.file == source.file_name());
    CHECK(record.line == source.line());
    CHECK(record.function == source.function_name());
  }
  CHECK(records.back().message == "threshold");
  CHECK(std::distance(std::filesystem::directory_iterator(fixture.root),
                      std::filesystem::directory_iterator{}) == 1);
}

TEST_CASE("Quent sink drains concurrent messages after swaps and context release",
          "[log][quent_sink]")
{
  sirius::test::scoped_recording_log_sink restore;
  telemetry_fixture fixture;
  auto sink = sirius::log::make_quent_sink(fixture.context);
  sink->set_level(level::trace);
  sirius::log::set_sink(sink);
  std::latch ready{4};
  std::atomic<bool> stop{false};
  std::vector<std::thread> threads;
  for (int index = 0; index < 4; ++index) {
    threads.emplace_back([&, index] {
      ready.count_down();
      for (int message = 0; message < 100; ++message) {
        sink->log(level::critical,
                  std::source_location::current(),
                  std::format("thread {} message {}", index, message));
      }
      while (!stop.load()) {
        sirius::log::get_sink()->log(level::critical, std::source_location::current(), "swapping");
      }
    });
  }
  ready.wait();
  for (int iteration = 0; iteration < 20; ++iteration) {
    sirius::log::set_sink(iteration % 2 ? sink : nullptr);
    sink->set_level(level::trace);
  }
  stop.store(true);
  for (auto& thread : threads) {
    thread.join();
  }
  sirius::log::set_sink(sink);
  fixture.context.reset();
  CHECK_FALSE(sink->should_log(level::critical));
  const auto records     = read_records(fixture.root);
  size_t thread_messages = 0;
  for (const auto& record : records) {
    if (record.message.starts_with("thread ")) { ++thread_messages; }
  }
  CHECK(thread_messages == 400);
}

TEST_CASE("Quent logging uses the most recently installed process-wide session",
          "[log][quent_sink]")
{
  sirius::test::scoped_recording_log_sink restore;
  telemetry_fixture first;
  telemetry_fixture second;
  auto first_sink  = sirius::log::make_quent_sink(first.context);
  auto second_sink = sirius::log::make_quent_sink(second.context);
  sirius::log::set_sink(first_sink);
  sirius::log::set_sink(second_sink);
  sirius::log::get_sink()->log(
    level::critical, std::source_location::current(), "selected session");
  first.context.reset();
  second.context.reset();
  const auto first_records  = read_records(first.root);
  const auto second_records = read_records(second.root);
  CHECK(first_records.size() == 1);
  REQUIRE(second_records.size() == 2);
  CHECK(second_records.back().message == "selected session");
}

TEST_CASE("Quent sink permits concurrent context destruction", "[log][quent_sink]")
{
  telemetry_fixture fixture;
  std::weak_ptr<const sirius::telemetry::telemetry_context> weak = fixture.context;
  auto sink = sirius::log::make_quent_sink(weak);
  std::latch ready{4};
  std::latch start{1};
  std::vector<std::thread> threads;
  for (int index = 0; index < 4; ++index) {
    threads.emplace_back([&] {
      ready.count_down();
      start.wait();
      for (int message = 0; message < 1000; ++message) {
        sink->log(level::critical, std::source_location::current(), "concurrent teardown");
      }
    });
  }
  ready.wait();
  start.count_down();
  fixture.context.reset();
  for (auto& thread : threads) {
    thread.join();
  }
  CHECK(weak.expired());
  CHECK_FALSE(sink->should_log(level::critical));
  const auto records = read_records(fixture.root);
  REQUIRE_FALSE(records.empty());
  CHECK(records.front().severity == "Initialized");
  for (size_t index = 1; index < records.size(); ++index) {
    CHECK(records[index].message == "concurrent teardown");
  }
}
