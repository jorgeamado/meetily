// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "fluid-helper",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(
            name: "fluid-helper",
            targets: ["FluidHelper"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/FluidInference/FluidAudio.git", exact: "0.18.2")
    ],
    targets: [
        .target(
            name: "FluidHelperCore",
            dependencies: [],
            path: "Sources/FluidHelperCore"
        ),
        .executableTarget(
            name: "FluidHelper",
            dependencies: [
                "FluidHelperCore",
                .product(name: "FluidAudio", package: "FluidAudio"),
            ],
            path: "Sources/FluidHelper"
        ),
        .testTarget(
            name: "FluidHelperCoreTests",
            dependencies: ["FluidHelperCore"],
            path: "Tests/FluidHelperCoreTests"
        ),
    ]
)
