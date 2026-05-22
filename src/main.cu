#include "matmul_kernel.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    const cudaError_t status__ = (expr);                                       \
    if (status__ != cudaSuccess) {                                             \
      std::ostringstream oss__;                                                \
      oss__ << "CUDA error: " << cudaGetErrorString(status__)                  \
            << " (" << static_cast<int>(status__) << ") at " << __FILE__       \
            << ":" << __LINE__;                                                \
      throw std::runtime_error(oss__.str());                                   \
    }                                                                          \
  } while (false)

struct ProblemSize {
  int m;
  int n;
  int k;
};

struct Options {
  std::vector<ProblemSize> sizes;
  std::vector<std::string> selected_kernels;
  int warmup_iterations = 3;
  int timed_iterations = 10;
  unsigned int seed = 1234;
  float tolerance = 1e-3f;
  std::optional<std::string> csv_path;
  std::optional<std::string> json_path;
};

struct TimingSummary {
  double min_ms = 0.0;
  double max_ms = 0.0;
  double mean_ms = 0.0;
  double median_ms = 0.0;
  double stddev_ms = 0.0;
};

struct BenchmarkResult {
  std::string kernel_name;
  ProblemSize size{};
  bool passed = false;
  float max_abs_error = 0.0f;
  TimingSummary timing{};
  double gflops_mean = 0.0;
  double gflops_best = 0.0;
  double effective_io_gbps = 0.0;
};

struct DeviceBuffer {
  float* ptr = nullptr;

  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t bytes) {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), bytes));
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept : ptr(other.ptr) { other.ptr = nullptr; }
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      reset();
      ptr = other.ptr;
      other.ptr = nullptr;
    }
    return *this;
  }

  ~DeviceBuffer() { reset(); }

  void reset() {
    if (ptr != nullptr) {
      cudaFree(ptr);
      ptr = nullptr;
    }
  }
};

struct CudaEvent {
  cudaEvent_t event = nullptr;

  CudaEvent() { CUDA_CHECK(cudaEventCreate(&event)); }
  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;

  ~CudaEvent() {
    if (event != nullptr) {
      cudaEventDestroy(event);
    }
  }
};

void print_usage(const char* program) {
  std::cerr
      << "Usage: " << program << " [options]\n"
      << "Options:\n"
      << "  --size MxNxK          Add one benchmark case. Repeatable.\n"
      << "  --kernel NAME         Benchmark one kernel by name. Repeatable. Default: all.\n"
      << "  --warmup N            Warmup launches before timing. Default: 3.\n"
      << "  --iters N             Timed launches per benchmark case. Default: 10.\n"
      << "  --seed N              RNG seed for host inputs. Default: 1234.\n"
      << "  --tol X               Max absolute error tolerance. Default: 1e-3.\n"
      << "  --csv PATH            Write CSV results.\n"
      << "  --json PATH           Write JSON results.\n"
      << "  --help                Show this message.\n"
      << "\n"
      << "Example:\n"
      << "  ./matmul_bench --size 512x768x256 --size 1024x1024x1024 "
         "--csv results.csv --json results.json\n";
}

ProblemSize parse_size(std::string_view text) {
  ProblemSize size{};
  std::string value(text);
  std::replace(value.begin(), value.end(), 'X', 'x');

  const auto first = value.find('x');
  const auto second = value.find('x', first == std::string::npos ? first : first + 1);
  if (first == std::string::npos || second == std::string::npos ||
      value.find('x', second + 1) != std::string::npos) {
    throw std::runtime_error("Invalid size. Expected MxNxK.");
  }

  size.m = std::stoi(value.substr(0, first));
  size.n = std::stoi(value.substr(first + 1, second - first - 1));
  size.k = std::stoi(value.substr(second + 1));
  if (size.m <= 0 || size.n <= 0 || size.k <= 0) {
    throw std::runtime_error("Matrix dimensions must be positive.");
  }
  return size;
}

Options parse_options(int argc, char** argv) {
  Options options;

  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i]);

    auto require_value = [&](std::string_view flag) -> std::string_view {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for " + std::string(flag));
      }
      ++i;
      return argv[i];
    };

    if (arg == "--size") {
      options.sizes.push_back(parse_size(require_value(arg)));
    } else if (arg == "--kernel") {
      options.selected_kernels.emplace_back(require_value(arg));
    } else if (arg == "--warmup") {
      options.warmup_iterations = std::stoi(std::string(require_value(arg)));
    } else if (arg == "--iters") {
      options.timed_iterations = std::stoi(std::string(require_value(arg)));
    } else if (arg == "--seed") {
      options.seed = static_cast<unsigned int>(std::stoul(std::string(require_value(arg))));
    } else if (arg == "--tol") {
      options.tolerance = std::stof(std::string(require_value(arg)));
    } else if (arg == "--csv") {
      options.csv_path = std::string(require_value(arg));
    } else if (arg == "--json") {
      options.json_path = std::string(require_value(arg));
    } else if (arg == "--help") {
      print_usage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown argument: " + std::string(arg));
    }
  }

  if (options.sizes.empty()) {
    options.sizes = {
        {256, 256, 256},
        {512, 512, 512},
        {1024, 1024, 1024},
    };
  }
  if (options.warmup_iterations < 0 || options.timed_iterations <= 0) {
    throw std::runtime_error("Warmup must be >= 0 and iters must be > 0.");
  }
  if (options.tolerance < 0.0f) {
    throw std::runtime_error("Tolerance must be non-negative.");
  }

  return options;
}

std::vector<float> random_matrix(std::size_t size, std::mt19937& rng) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> out(size);
  for (float& value : out) {
    value = dist(rng);
  }
  return out;
}

std::vector<float> cpu_matmul(
    const std::vector<float>& a,
    const std::vector<float>& b,
    int m,
    int n,
    int k) {
  std::vector<float> c(static_cast<std::size_t>(m) * n, 0.0f);
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      float acc = 0.0f;
      for (int inner = 0; inner < k; ++inner) {
        acc += a[static_cast<std::size_t>(row) * k + inner] *
               b[static_cast<std::size_t>(inner) * n + col];
      }
      c[static_cast<std::size_t>(row) * n + col] = acc;
    }
  }
  return c;
}

float max_abs_error(const std::vector<float>& expected, const std::vector<float>& actual) {
  float max_error = 0.0f;
  for (std::size_t i = 0; i < expected.size(); ++i) {
    max_error = std::max(max_error, std::fabs(expected[i] - actual[i]));
  }
  return max_error;
}

TimingSummary summarize(std::vector<float> samples_ms) {
  std::sort(samples_ms.begin(), samples_ms.end());

  TimingSummary summary;
  summary.min_ms = samples_ms.front();
  summary.max_ms = samples_ms.back();

  const double sum = std::accumulate(samples_ms.begin(), samples_ms.end(), 0.0);
  summary.mean_ms = sum / static_cast<double>(samples_ms.size());

  const std::size_t mid = samples_ms.size() / 2;
  if (samples_ms.size() % 2 == 0) {
    summary.median_ms = 0.5 * (samples_ms[mid - 1] + samples_ms[mid]);
  } else {
    summary.median_ms = samples_ms[mid];
  }

  double variance = 0.0;
  for (const double sample : samples_ms) {
    const double delta = sample - summary.mean_ms;
    variance += delta * delta;
  }
  variance /= static_cast<double>(samples_ms.size());
  summary.stddev_ms = std::sqrt(variance);

  return summary;
}

std::vector<MatmulKernelSpec> select_kernels(const Options& options) {
  const auto& all = registered_matmul_kernels();
  if (options.selected_kernels.empty()) {
    return std::vector<MatmulKernelSpec>(all.begin(), all.end());
  }

  std::vector<MatmulKernelSpec> selected;
  for (const std::string& wanted : options.selected_kernels) {
    const auto it = std::find_if(
        all.begin(), all.end(), [&](const MatmulKernelSpec& spec) { return spec.name == wanted; });
    if (it == all.end()) {
      throw std::runtime_error("Unknown kernel: " + wanted);
    }
    selected.push_back(*it);
  }
  return selected;
}

BenchmarkResult benchmark_kernel(
    const MatmulKernelSpec& kernel,
    const ProblemSize& size,
    const Options& options,
    std::mt19937& rng) {
  const std::size_t a_elems = static_cast<std::size_t>(size.m) * size.k;
  const std::size_t b_elems = static_cast<std::size_t>(size.k) * size.n;
  const std::size_t c_elems = static_cast<std::size_t>(size.m) * size.n;
  const std::size_t a_bytes = a_elems * sizeof(float);
  const std::size_t b_bytes = b_elems * sizeof(float);
  const std::size_t c_bytes = c_elems * sizeof(float);

  const std::vector<float> host_a = random_matrix(a_elems, rng);
  const std::vector<float> host_b = random_matrix(b_elems, rng);
  const std::vector<float> expected = cpu_matmul(host_a, host_b, size.m, size.n, size.k);
  std::vector<float> host_c(c_elems, 0.0f);

  DeviceBuffer device_a(a_bytes);
  DeviceBuffer device_b(b_bytes);
  DeviceBuffer device_c(c_bytes);

  CUDA_CHECK(cudaMemcpy(device_a.ptr, host_a.data(), a_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_b.ptr, host_b.data(), b_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(device_c.ptr, 0, c_bytes));

  kernel.launch(device_a.ptr, device_b.ptr, device_c.ptr, size.m, size.n, size.k, nullptr);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(host_c.data(), device_c.ptr, c_bytes, cudaMemcpyDeviceToHost));
  const float error = max_abs_error(expected, host_c);
  const bool passed = error <= options.tolerance;

  for (int i = 0; i < options.warmup_iterations; ++i) {
    kernel.launch(device_a.ptr, device_b.ptr, device_c.ptr, size.m, size.n, size.k, nullptr);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CudaEvent start;
  CudaEvent stop;
  std::vector<float> samples_ms;
  samples_ms.reserve(static_cast<std::size_t>(options.timed_iterations));

  for (int i = 0; i < options.timed_iterations; ++i) {
    CUDA_CHECK(cudaEventRecord(start.event));
    kernel.launch(device_a.ptr, device_b.ptr, device_c.ptr, size.m, size.n, size.k, nullptr);
    CUDA_CHECK(cudaEventRecord(stop.event));
    CUDA_CHECK(cudaEventSynchronize(stop.event));
    CUDA_CHECK(cudaGetLastError());

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start.event, stop.event));
    samples_ms.push_back(elapsed_ms);
  }

  const TimingSummary timing = summarize(samples_ms);
  const double flops = 2.0 * static_cast<double>(size.m) * size.n * size.k;
  const double bytes = static_cast<double>(a_bytes + b_bytes + c_bytes);

  BenchmarkResult result;
  result.kernel_name = std::string(kernel.name);
  result.size = size;
  result.passed = passed;
  result.max_abs_error = error;
  result.timing = timing;
  result.gflops_mean = flops / (timing.mean_ms * 1.0e6);
  result.gflops_best = flops / (timing.min_ms * 1.0e6);
  result.effective_io_gbps = bytes / (timing.mean_ms * 1.0e6);
  return result;
}

std::string escape_json(std::string_view text) {
  std::string out;
  out.reserve(text.size());
  for (const char ch : text) {
    switch (ch) {
      case '\\':
        out += "\\\\";
        break;
      case '"':
        out += "\\\"";
        break;
      case '\n':
        out += "\\n";
        break;
      default:
        out += ch;
        break;
    }
  }
  return out;
}

void write_csv(
    const std::string& path,
    const std::vector<BenchmarkResult>& results,
    const Options& options) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Failed to open CSV output: " + path);
  }

  out << "kernel,m,n,k,passed,max_abs_error,warmup_iterations,timed_iterations,"
         "min_ms,max_ms,mean_ms,median_ms,stddev_ms,gflops_mean,gflops_best,"
         "effective_io_gbps\n";

  out << std::fixed << std::setprecision(6);
  for (const BenchmarkResult& result : results) {
    out << result.kernel_name << ','
        << result.size.m << ','
        << result.size.n << ','
        << result.size.k << ','
        << (result.passed ? "true" : "false") << ','
        << result.max_abs_error << ','
        << options.warmup_iterations << ','
        << options.timed_iterations << ','
        << result.timing.min_ms << ','
        << result.timing.max_ms << ','
        << result.timing.mean_ms << ','
        << result.timing.median_ms << ','
        << result.timing.stddev_ms << ','
        << result.gflops_mean << ','
        << result.gflops_best << ','
        << result.effective_io_gbps << '\n';
  }
}

void write_json(
    const std::string& path,
    const std::vector<BenchmarkResult>& results,
    const Options& options,
    const cudaDeviceProp& prop) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Failed to open JSON output: " + path);
  }

  out << std::fixed << std::setprecision(6);
  out << "{\n"
      << "  \"device\": {\n"
      << "    \"name\": \"" << escape_json(prop.name) << "\",\n"
      << "    \"sm_count\": " << prop.multiProcessorCount << ",\n"
      << "    \"compute_capability\": \"" << prop.major << "." << prop.minor << "\"\n"
      << "  },\n"
      << "  \"config\": {\n"
      << "    \"warmup_iterations\": " << options.warmup_iterations << ",\n"
      << "    \"timed_iterations\": " << options.timed_iterations << ",\n"
      << "    \"seed\": " << options.seed << ",\n"
      << "    \"tolerance\": " << options.tolerance << "\n"
      << "  },\n"
      << "  \"results\": [\n";

  for (std::size_t i = 0; i < results.size(); ++i) {
    const BenchmarkResult& result = results[i];
    out << "    {\n"
        << "      \"kernel\": \"" << escape_json(result.kernel_name) << "\",\n"
        << "      \"m\": " << result.size.m << ",\n"
        << "      \"n\": " << result.size.n << ",\n"
        << "      \"k\": " << result.size.k << ",\n"
        << "      \"passed\": " << (result.passed ? "true" : "false") << ",\n"
        << "      \"max_abs_error\": " << result.max_abs_error << ",\n"
        << "      \"min_ms\": " << result.timing.min_ms << ",\n"
        << "      \"max_ms\": " << result.timing.max_ms << ",\n"
        << "      \"mean_ms\": " << result.timing.mean_ms << ",\n"
        << "      \"median_ms\": " << result.timing.median_ms << ",\n"
        << "      \"stddev_ms\": " << result.timing.stddev_ms << ",\n"
        << "      \"gflops_mean\": " << result.gflops_mean << ",\n"
        << "      \"gflops_best\": " << result.gflops_best << ",\n"
        << "      \"effective_io_gbps\": " << result.effective_io_gbps << "\n"
        << "    }";
    if (i + 1 != results.size()) {
      out << ',';
    }
    out << '\n';
  }

  out << "  ]\n"
      << "}\n";
}

void print_summary(
    const std::vector<BenchmarkResult>& results,
    int warmup_iterations,
    int timed_iterations,
    const cudaDeviceProp& prop) {
  std::cout << "Device: " << prop.name << " (cc " << prop.major << '.' << prop.minor << ")\n";
  std::cout << "Warmup iterations: " << warmup_iterations
            << ", timed iterations: " << timed_iterations << "\n\n";

  std::cout << std::left << std::setw(14) << "kernel"
            << std::setw(14) << "size(M,N,K)"
            << std::setw(10) << "pass"
            << std::setw(14) << "mean_ms"
            << std::setw(14) << "median_ms"
            << std::setw(14) << "best_ms"
            << std::setw(14) << "GF/s mean"
            << std::setw(14) << "GF/s best"
            << "max_err\n";

  std::cout << std::fixed << std::setprecision(4);
  for (const BenchmarkResult& result : results) {
    std::ostringstream size_text;
    size_text << result.size.m << 'x' << result.size.n << 'x' << result.size.k;
    std::cout << std::left << std::setw(14) << result.kernel_name
              << std::setw(14) << size_text.str()
              << std::setw(10) << (result.passed ? "yes" : "no")
              << std::setw(14) << result.timing.mean_ms
              << std::setw(14) << result.timing.median_ms
              << std::setw(14) << result.timing.min_ms
              << std::setw(14) << result.gflops_mean
              << std::setw(14) << result.gflops_best
              << result.max_abs_error << '\n';
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    const std::vector<MatmulKernelSpec> kernels = select_kernels(options);

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::mt19937 rng(options.seed);
    std::vector<BenchmarkResult> results;
    results.reserve(options.sizes.size() * kernels.size());

    for (const ProblemSize& size : options.sizes) {
      for (const MatmulKernelSpec& kernel : kernels) {
        results.push_back(benchmark_kernel(kernel, size, options, rng));
      }
    }

    print_summary(results, options.warmup_iterations, options.timed_iterations, prop);

    if (options.csv_path.has_value()) {
      write_csv(*options.csv_path, results, options);
    }
    if (options.json_path.has_value()) {
      write_json(*options.json_path, results, options, prop);
    }

    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << '\n';
    return 1;
  }
}
