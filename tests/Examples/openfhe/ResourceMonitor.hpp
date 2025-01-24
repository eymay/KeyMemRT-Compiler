#include <omp.h>
#include <sys/resource.h>

#include <atomic>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <shared_mutex>
#include <sstream>
#include <thread>
#include <vector>

class ResourceMonitor {
 private:
  struct PageFaultMetrics {
    long minor_faults;  // Page faults satisfied by kernel page cache
    long major_faults;  // Page faults requiring disk I/O
    long minor_faults_children;
    long major_faults_children;
  };
  struct MemoryMetrics {
    double ram_total;          // Total RAM in GB
    double ram_used;           // Used RAM in GB
    double ram_free;           // Free RAM in GB
    double ram_cached;         // Cached RAM in GB
    double ram_buffers;        // Buffer RAM in GB
    double swap_total;         // Total swap in GB
    double swap_used;          // Used swap in GB
    double swap_free;          // Free swap in GB
    double swap_cached;        // Swap cached in GB
    unsigned long vm_pgpgin;   // Pages paged in
    unsigned long vm_pgpgout;  // Pages paged out
    unsigned long vm_pswpin;   // Pages swapped in
    unsigned long vm_pswpout;  // Pages swapped out
    PageFaultMetrics page_faults;
  };

  struct Sample {
    double timestamp;
    std::vector<double> cpu_usage;
    MemoryMetrics memory;
  };

  struct RotationTiming {
    int target;
    int thread_id;
    double start_time;  // Time since start
    double end_time;    // Time since start
    PageFaultMetrics page_faults_start;
    PageFaultMetrics page_faults_end;
  };

  std::vector<Sample> samples;
  std::vector<RotationTiming> rotation_timings;
  mutable std::shared_mutex timing_mutex;
  mutable std::mutex samples_mutex;

  std::atomic<bool> should_run{true};
  std::thread monitor_thread;
  const std::chrono::milliseconds sample_interval{10};

  // Single time base for all measurements
  std::chrono::steady_clock::time_point start_time;

  PageFaultMetrics get_page_faults() {
    struct rusage usage;
    PageFaultMetrics metrics{};

    if (getrusage(RUSAGE_SELF, &usage) == 0) {
      metrics.minor_faults = usage.ru_minflt;
      metrics.major_faults = usage.ru_majflt;
    }

    if (getrusage(RUSAGE_CHILDREN, &usage) == 0) {
      metrics.minor_faults_children = usage.ru_minflt;
      metrics.major_faults_children = usage.ru_majflt;
    }

    return metrics;
  }

  double get_elapsed_time() const {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                         start_time)
        .count();
  }

  static std::vector<double> get_cpu_usage() {
    static std::vector<unsigned long long> last_idle, last_total;
    std::vector<double> usage;
    std::ifstream stat_file("/proc/stat");
    std::string line;
    int core = -1;  // First line is aggregate, skip it

    while (std::getline(stat_file, line) && line.compare(0, 3, "cpu") == 0) {
      if (core++ < 0) continue;

      std::istringstream ss(line);
      std::string cpu;
      unsigned long long user, nice, system, idle, iowait, irq, softirq, steal,
          guest, guest_nice;
      ss >> cpu >> user >> nice >> system >> idle >> iowait >> irq >> softirq >>
          steal >> guest >> guest_nice;

      unsigned long long total =
          user + nice + system + idle + iowait + irq + softirq + steal;

      if (last_idle.size() <= static_cast<size_t>(core)) {
        last_idle.push_back(idle);
        last_total.push_back(total);
        usage.push_back(0.0);
      } else {
        double idle_delta = idle - last_idle[core];
        double total_delta = total - last_total[core];
        usage.push_back(100.0 * (1.0 - idle_delta / total_delta));

        last_idle[core] = idle;
        last_total[core] = total;
      }
    }
    return usage;
  }

  MemoryMetrics get_memory_metrics() {
    MemoryMetrics metrics{};

    // Read /proc/meminfo for memory stats
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    while (std::getline(meminfo, line)) {
      std::istringstream ss(line);
      std::string key;
      unsigned long value;
      ss >> key >> value;

      if (key == "MemTotal:")
        metrics.ram_total = value / 1024.0 / 1024.0;
      else if (key == "MemFree:")
        metrics.ram_free = value / 1024.0 / 1024.0;
      else if (key == "Cached:")
        metrics.ram_cached = value / 1024.0 / 1024.0;
      else if (key == "Buffers:")
        metrics.ram_buffers = value / 1024.0 / 1024.0;
      else if (key == "SwapTotal:")
        metrics.swap_total = value / 1024.0 / 1024.0;
      else if (key == "SwapFree:")
        metrics.swap_free = value / 1024.0 / 1024.0;
      else if (key == "SwapCached:")
        metrics.swap_cached = value / 1024.0 / 1024.0;
    }

    // Read /proc/vmstat for page and swap statistics
    std::ifstream vmstat("/proc/vmstat");
    while (std::getline(vmstat, line)) {
      std::istringstream ss(line);
      std::string key;
      unsigned long value;
      ss >> key >> value;

      if (key == "pgpgin")
        metrics.vm_pgpgin = value;
      else if (key == "pgpgout")
        metrics.vm_pgpgout = value;
      else if (key == "pswpin")
        metrics.vm_pswpin = value;
      else if (key == "pswpout")
        metrics.vm_pswpout = value;
    }

    // Get page fault statistics
    metrics.page_faults = get_page_faults();

    // Calculate derived values
    metrics.ram_used = metrics.ram_total - metrics.ram_free -
                       metrics.ram_cached - metrics.ram_buffers;
    metrics.swap_used = metrics.swap_total - metrics.swap_free;

    return metrics;
  }

 public:
  void start() {
    should_run = true;
    // Initialize the time base when monitoring starts
    start_time = std::chrono::steady_clock::now();

    monitor_thread = std::thread([this]() {
      while (should_run) {
        Sample sample;
        sample.timestamp = get_elapsed_time();
        sample.cpu_usage = get_cpu_usage();
        sample.memory = get_memory_metrics();

        {
          std::lock_guard<std::mutex> lock(samples_mutex);
          samples.push_back(sample);
        }

        std::this_thread::sleep_for(sample_interval);
      }
    });
  }

  void stop() {
    should_run = false;
    if (monitor_thread.joinable()) {
      monitor_thread.join();
    }
  }

  void record_rotation_start(int target) {
    double current_time = get_elapsed_time();

    RotationTiming timing;
    timing.target = target;
    timing.thread_id = omp_get_thread_num();
    timing.start_time = current_time;
    timing.end_time = -1;
    timing.page_faults_start = get_page_faults();

    {
      std::unique_lock<std::shared_mutex> lock(timing_mutex);
      rotation_timings.push_back(timing);
    }
  }

  void record_rotation_end(int target) {
    double current_time = get_elapsed_time();
    int thread_id = omp_get_thread_num();
    auto page_faults_end = get_page_faults();

    std::unique_lock<std::shared_mutex> lock(timing_mutex);
    // Find the matching start timing for this thread and target
    for (auto it = rotation_timings.rbegin(); it != rotation_timings.rend();
         ++it) {
      if (it->target == target && it->thread_id == thread_id &&
          it->end_time == -1) {
        it->end_time = current_time;
        it->page_faults_end = page_faults_end;
        break;
      }
    }
  }

  void save_to_file(const std::string &base_filename) const {
    // Save resource usage data
    std::string resource_filename = base_filename + "_resources.csv";
    std::ofstream file(resource_filename);
    file << std::fixed << std::setprecision(6);

    // Enhanced header with new memory metrics
    file << "Time,RAM_Used_GB,RAM_Free_GB,RAM_Cached_GB,RAM_Buffers_GB,RAM_"
            "Total_GB,"
         << "Swap_Used_GB,Swap_Free_GB,Swap_Cached_GB,Swap_Total_GB,"
         << "Pages_In,Pages_Out,Swap_Pages_In,Swap_Pages_Out,"
         << "Minor_Faults,Major_Faults,Minor_Faults_Children,Major_Faults_"
            "Children";

    for (size_t i = 0; i < samples[0].cpu_usage.size(); ++i) {
      file << ",CPU" << i;
    }
    file << "\n";

    {
      std::lock_guard<std::mutex> lock(samples_mutex);
      for (const auto &sample : samples) {
        const auto &m = sample.memory;
        const auto &pf = m.page_faults;
        file << sample.timestamp << "," << m.ram_used << "," << m.ram_free
             << "," << m.ram_cached << "," << m.ram_buffers << ","
             << m.ram_total << "," << m.swap_used << "," << m.swap_free << ","
             << m.swap_cached << "," << m.swap_total << "," << m.vm_pgpgin
             << "," << m.vm_pgpgout << "," << m.vm_pswpin << "," << m.vm_pswpout
             << "," << pf.minor_faults << "," << pf.major_faults << ","
             << pf.minor_faults_children << "," << pf.major_faults_children;

        for (double cpu : sample.cpu_usage) {
          file << "," << cpu;
        }
        file << "\n";
      }
    }

    // Save rotation timing data
    std::string timing_filename = base_filename + "_rotations.csv";
    std::ofstream timing_file(timing_filename);
    timing_file << std::fixed << std::setprecision(6);

    timing_file << "Target,ThreadID,StartTime,EndTime,Duration,"
                << "Minor_Faults,Major_Faults,Minor_Faults_Children,Major_"
                   "Faults_Children\n";

    std::shared_lock<std::shared_mutex> lock(timing_mutex);
    for (const auto &timing : rotation_timings) {
      double duration = timing.end_time - timing.start_time;
      auto minor_faults = timing.page_faults_end.minor_faults -
                          timing.page_faults_start.minor_faults;
      auto major_faults = timing.page_faults_end.major_faults -
                          timing.page_faults_start.major_faults;
      auto minor_faults_children =
          timing.page_faults_end.minor_faults_children -
          timing.page_faults_start.minor_faults_children;
      auto major_faults_children =
          timing.page_faults_end.major_faults_children -
          timing.page_faults_start.major_faults_children;

      timing_file << timing.target << "," << timing.thread_id << ","
                  << timing.start_time << "," << timing.end_time << ","
                  << duration << "," << minor_faults << "," << major_faults
                  << "," << minor_faults_children << ","
                  << major_faults_children << "\n";
    }
  }
};
