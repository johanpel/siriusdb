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

#include "log/quent_sink.hpp"

#include "telemetry/telemetry_context.hpp"

#include <atomic>
#include <stdexcept>
#include <string>

namespace sirius::log {
namespace {

class quent_sink final : public sink {
 public:
  explicit quent_sink(std::weak_ptr<const telemetry::telemetry_context> context)
    : context_(std::move(context))
  {
    auto live_context = context_.lock();
    if (!live_context || !live_context->enabled()) {
      throw std::invalid_argument("Quent logging requires an enabled engine telemetry session");
    }
  }

  void set_level(level lvl) override { level_.store(lvl, std::memory_order_relaxed); }

  [[nodiscard]] bool should_log(level lvl) const override
  {
    return lvl != level::off && lvl >= level_.load(std::memory_order_relaxed) &&
           !context_.expired();
  }

  void log(level lvl, const std::source_location& location, std::string_view message) override
  {
    if (!should_log(lvl)) { return; }
    auto context = context_.lock();
    if (!context) { return; }
    const auto& handle = context->log_handle();
    switch (lvl) {
      case level::trace:
        handle.trace({.message  = std::string(message),
                      .file     = location.file_name(),
                      .line     = location.line(),
                      .function = location.function_name()});
        break;
      case level::debug:
        handle.debug({.message  = std::string(message),
                      .file     = location.file_name(),
                      .line     = location.line(),
                      .function = location.function_name()});
        break;
      case level::info:
        handle.info({.message  = std::string(message),
                     .file     = location.file_name(),
                     .line     = location.line(),
                     .function = location.function_name()});
        break;
      case level::warn:
        handle.warn({.message  = std::string(message),
                     .file     = location.file_name(),
                     .line     = location.line(),
                     .function = location.function_name()});
        break;
      case level::error:
        handle.error({.message  = std::string(message),
                      .file     = location.file_name(),
                      .line     = location.line(),
                      .function = location.function_name()});
        break;
      case level::critical:
        handle.critical({.message  = std::string(message),
                         .file     = location.file_name(),
                         .line     = location.line(),
                         .function = location.function_name()});
        break;
      case level::off: break;
    }
  }

  bool flush() override { return false; }

 private:
  std::weak_ptr<const telemetry::telemetry_context> context_;
  std::atomic<level> level_{level::info};
};

}  // namespace

std::shared_ptr<sink> make_quent_sink(std::weak_ptr<const telemetry::telemetry_context> context)
{
  return std::make_shared<quent_sink>(std::move(context));
}

}  // namespace sirius::log
