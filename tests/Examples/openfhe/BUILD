package(default_applicable_licenses = ["@heir//:license"])

package_group(
    name = "all_packages",
    packages = ["//..."],
)

filegroup(
    name = "common_headers",
    srcs = [
        "KeyMemRT.hpp",
        "ResourceMonitor.hpp",
    ],
    visibility = [":all_packages"],
)
