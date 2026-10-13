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

#pragma once

#include "log/sink.hpp"

#include <memory>

namespace sirius::telemetry {
class telemetry_context;
}

namespace sirius::log {

/// Creates a sink that exports log messages through enabled Quent telemetry.
///
/// Keep `context` alive until work has finished and logging threads have been
/// joined. The sink holds a weak reference and temporarily retains the context
/// during each emission. After the context expires, `sink::should_log()`
/// returns false and messages are immediately discarded.
///
/// Messages queued before context destruction are exported during shutdown.
/// Explicit flushing is not supported, so `sink::flush()` always returns false.
///
/// Throws `std::invalid_argument` if `context` is empty or expired, or if
/// telemetry is disabled.
std::shared_ptr<sink> make_quent_sink(std::weak_ptr<const telemetry::telemetry_context> context);

}  // namespace sirius::log
