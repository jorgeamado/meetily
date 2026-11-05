import Foundation
import os

/// Output returned by a `Transcriber` for a single audio file.
public struct TranscriptionOutput: Sendable, Equatable {
    public let text: String
    public let confidence: Double
    public let duration: Double
    public let processingSecs: Double
    public let words: [WordTimingItem]
    public let tokens: [TokenTimingItem]

    public init(
        text: String,
        confidence: Double,
        duration: Double,
        processingSecs: Double,
        words: [WordTimingItem],
        tokens: [TokenTimingItem]
    ) {
        self.text = text
        self.confidence = confidence
        self.duration = duration
        self.processingSecs = processingSecs
        self.words = words
        self.tokens = tokens
    }
}

/// Status information returned by a `Transcriber` for a model version.
public struct ModelStatusOutput: Sendable, Equatable {
    public let downloaded: Bool
    public let path: String

    public init(downloaded: Bool, path: String) {
        self.downloaded = downloaded
        self.path = path
    }
}

/// Abstraction over the speech-to-text backend so the JSON protocol loop can be tested without CoreML/FluidAudio.
public protocol Transcriber: Sendable {
    /// Download (if missing) and load the specified model version from `<modelsDir>/<repo.folderName>`.
    /// Returns the elapsed load time in seconds.
    func load(
        version: ModelVersion,
        modelsDir: String,
        onProgress: @escaping @Sendable (ProgressStage, Double) -> Void
    ) async throws -> Double

    /// Check whether the specified model version is downloaded and load-ready at `<modelsDir>/<repo.folderName>`
    /// without downloading or loading any models.
    func modelStatus(
        version: ModelVersion,
        modelsDir: String
    ) async throws -> ModelStatusOutput

    /// Transcribe the WAV file at `wavPath`.
    func transcribe(
        wavPath: String,
        includeTokens: Bool
    ) async throws -> TranscriptionOutput
}

/// Thread-safe progress normalizer for a single `load` request.
///
/// Guarantees that emitted progress fractions are:
/// - monotonic non-decreasing within each stage (`download` and `compile`), with `compile` starting at `0.0` and never switching back to `download`,
/// - rate-limited to steps of at least `minEmissionDelta` (0.01),
/// - capped at `inProgressCap` (0.99) while downloading or compiling is still in progress, and
/// - completed with a final `fraction: 1.0` on `compile` only when `complete()` is called right before `loaded`.
public final class LoadProgressBridge: Sendable {
    public static let inProgressCap: Double = 0.99
    public static let minEmissionDelta: Double = 0.01
    private static let cachedCompileCallbackCount: Double = 8.0
    private static let coldCompileCallbackCount: Double = 16.0

    private struct State: Sendable {
        var hadDownload = false
        var compileSteps: Int = 0
        var lastStage: ProgressStage = .download
        var lastEmittedFraction: Double = -1.0
        var isCompleted = false
    }

    private let state = OSAllocatedUnfairLock(initialState: State())
    private let onProgress: @Sendable (ProgressStage, Double) -> Void

    public init(onProgress: @escaping @Sendable (ProgressStage, Double) -> Void) {
        self.onProgress = onProgress
    }

    /// Handle FluidAudio's `.listing` phase at the start of a repository download.
    public func handleListing() {
        let emission = state.withLock { current -> (ProgressStage, Double)? in
            guard !current.isCompleted, current.lastStage != .compile else { return nil }
            current.hadDownload = true
            guard current.lastEmittedFraction < 0 else { return nil }
            current.lastStage = .download
            current.lastEmittedFraction = 0.0
            return (.download, 0.0)
        }
        guard let emission else { return }
        onProgress(emission.0, emission.1)
    }

    /// Handle FluidAudio's `.downloading` phase.
    public func handleDownloading(fractionCompleted: Double, totalFiles: Int) {
        // ModelHub emits .downloading(completedFiles: 0, totalFiles: 0) on local cache hits; ignore those.
        guard totalFiles > 0 else { return }
        let emission = state.withLock { current -> (ProgressStage, Double)? in
            guard !current.isCompleted, current.lastStage != .compile else { return nil }
            current.hadDownload = true
            return Self.recordInProgressLocked(&current, stage: .download, rawFraction: fractionCompleted)
        }
        guard let emission else { return }
        onProgress(emission.0, emission.1)
    }

    /// Handle FluidAudio's `.compiling(modelName:)` phase across per-component `ModelHub.loadModels` calls.
    public func handleCompiling(modelName: String) {
        _ = modelName
        let emission = state.withLock { current -> (ProgressStage, Double)? in
            guard !current.isCompleted else { return nil }
            current.compileSteps += 1
            let expectedCallbacks =
                current.hadDownload
                ? Self.coldCompileCallbackCount
                : Self.cachedCompileCallbackCount
            let rawFraction = Double(current.compileSteps) / expectedCallbacks
            return Self.recordInProgressLocked(&current, stage: .compile, rawFraction: rawFraction)
        }
        guard let emission else { return }
        onProgress(emission.0, emission.1)
    }

    /// Normalize and rate-limit an arbitrary in-progress `(stage, fraction)` update.
    public func report(stage: ProgressStage, fraction: Double) {
        let emission = state.withLock { current -> (ProgressStage, Double)? in
            guard !current.isCompleted else { return nil }
            if stage == .download {
                guard current.lastStage != .compile else { return nil }
                current.hadDownload = true
            }
            return Self.recordInProgressLocked(&current, stage: stage, rawFraction: fraction)
        }
        guard let emission else { return }
        onProgress(emission.0, emission.1)
    }

    /// Emit the final `fraction: 1.0` progress event right before `loaded`.
    public func complete(stage: ProgressStage = .compile) {
        let emission = state.withLock { current -> (ProgressStage, Double)? in
            guard !current.isCompleted else { return nil }
            current.isCompleted = true
            let resolvedStage: ProgressStage = current.lastStage == .compile ? .compile : stage
            current.lastStage = resolvedStage
            guard current.lastEmittedFraction < 1.0 else { return nil }
            current.lastEmittedFraction = 1.0
            return (resolvedStage, 1.0)
        }
        guard let emission else { return }
        onProgress(emission.0, emission.1)
    }

    private static func recordInProgressLocked(
        _ current: inout State,
        stage: ProgressStage,
        rawFraction: Double
    ) -> (ProgressStage, Double)? {
        if current.lastStage == .compile && stage == .download {
            return nil
        }
        if current.lastStage != stage {
            current.lastStage = stage
            current.lastEmittedFraction = -1.0
        }
        let clamped = min(inProgressCap, max(0.0, rawFraction))
        let nextFraction = max(current.lastEmittedFraction, clamped)
        let isFirstEmission = current.lastEmittedFraction < 0
        let delta = nextFraction - current.lastEmittedFraction
        guard isFirstEmission || delta >= minEmissionDelta else {
            return nil
        }
        if !isFirstEmission && nextFraction == current.lastEmittedFraction {
            return nil
        }
        current.lastStage = stage
        current.lastEmittedFraction = nextFraction
        return (stage, nextFraction)
    }
}

public typealias LoadProgressNormalizer = LoadProgressBridge

/// Thread-safe tracker for request activity and idle timeout expiration.
public final class ActivityTracker: Sendable {
    private struct State: Sendable {
        var inFlight: Bool
        var lastActivity: ContinuousClock.Instant
    }

    private let state: OSAllocatedUnfairLock<State>

    public init(now: ContinuousClock.Instant = .now) {
        self.state = OSAllocatedUnfairLock(initialState: State(inFlight: false, lastActivity: now))
    }

    public func beginRequest(now: ContinuousClock.Instant = .now) {
        state.withLock { current in
            current.inFlight = true
            current.lastActivity = now
        }
    }

    public func endRequest(now: ContinuousClock.Instant = .now) {
        state.withLock { current in
            current.inFlight = false
            current.lastActivity = now
        }
    }

    public func touch(now: ContinuousClock.Instant = .now) {
        state.withLock { current in
            current.lastActivity = now
        }
    }

    public func isIdleExpired(
        timeoutSeconds: TimeInterval,
        now: ContinuousClock.Instant = .now
    ) -> Bool {
        state.withLock { current in
            guard !current.inFlight else { return false }
            let elapsed = now - current.lastActivity
            let threshold = Duration.seconds(timeoutSeconds)
            return elapsed >= threshold
        }
    }
}

/// Sequential JSON-lines protocol session backed by an injected `Transcriber`.
public actor ProtocolSession {
    private let transcriber: any Transcriber
    private var loadedVersion: ModelVersion?

    public init(transcriber: any Transcriber) {
        self.transcriber = transcriber
    }

    /// Process a single line from `stdin`.
    /// Calls `emit` with each outbound `Response` (including any intermediate `progress` responses).
    /// Returns `true` to continue processing subsequent lines, or `false` when `shutdown` was handled.
    public func handleLine(
        _ rawLine: String,
        emit: @escaping @Sendable (Response) -> Void
    ) async -> Bool {
        let trimmed = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            return true
        }

        let request: Request
        do {
            request = try ProtocolCodec.decodeRequest(from: trimmed)
        } catch {
            emit(.error(id: nil, message: ProtocolCodec.describeError(error)))
            return true
        }

        switch request {
        case .ping:
            emit(.pong)
            return true

        case .shutdown:
            emit(.goodbye)
            return false

        case .load(let version, let modelsDir):
            await handleLoad(version: version, modelsDir: modelsDir, emit: emit)
            return true

        case .status(let version, let modelsDir):
            await handleStatus(version: version, modelsDir: modelsDir, emit: emit)
            return true

        case .transcribe(let id, let wavPath, let includeTokens):
            await handleTranscribe(id: id, wavPath: wavPath, includeTokens: includeTokens, emit: emit)
            return true
        }
    }

    /// Run a sequence of JSON input lines through the session and return the encoded JSON output lines.
    public func runLines<S: Sequence>(_ lines: S) async -> [String] where S.Element == String {
        let outputLines = OSAllocatedUnfairLock(initialState: [String]())
        let emitLine: @Sendable (Response) -> Void = { response in
            guard let json = try? ProtocolCodec.encodeResponse(response) else { return }
            outputLines.withLock { $0.append(json) }
        }

        for line in lines {
            let shouldContinue = await handleLine(line, emit: emitLine)
            guard shouldContinue else { break }
        }

        return outputLines.withLock { $0 }
    }

    private func handleStatus(
        version: ModelVersion,
        modelsDir: String,
        emit: @Sendable (Response) -> Void
    ) async {
        do {
            let status = try await transcriber.modelStatus(version: version, modelsDir: modelsDir)
            emit(.status(version: version, downloaded: status.downloaded, path: status.path))
        } catch {
            emit(.error(id: nil, message: ProtocolCodec.describeError(error)))
        }
    }

    private func handleLoad(
        version: ModelVersion,
        modelsDir: String,
        emit: @escaping @Sendable (Response) -> Void
    ) async {
        if loadedVersion == version {
            emit(.loaded(version: version, loadSecs: 0.0))
            return
        }

        loadedVersion = nil
        do {
            let loadSecs = try await transcriber.load(
                version: version,
                modelsDir: modelsDir,
                onProgress: { stage, fraction in
                    let clamped = min(1.0, max(0.0, fraction))
                    emit(.progress(stage: stage, fraction: clamped))
                }
            )
            loadedVersion = version
            emit(.loaded(version: version, loadSecs: loadSecs))
        } catch {
            emit(.error(id: nil, message: ProtocolCodec.describeError(error)))
        }
    }

    private func handleTranscribe(
        id: Int,
        wavPath: String,
        includeTokens: Bool,
        emit: @Sendable (Response) -> Void
    ) async {
        guard loadedVersion != nil else {
            emit(.error(id: id, message: ProtocolCodec.describeError(ProtocolError.modelNotLoaded)))
            return
        }

        do {
            let output = try await transcriber.transcribe(wavPath: wavPath, includeTokens: includeTokens)
            let tokens: [TokenTimingItem]? = includeTokens ? output.tokens : nil
            let payload = ResultPayload(
                id: id,
                text: output.text,
                confidence: output.confidence,
                duration: output.duration,
                processingSecs: output.processingSecs,
                words: output.words,
                tokens: tokens
            )
            emit(.result(payload))
        } catch {
            emit(.error(id: id, message: ProtocolCodec.describeError(error)))
        }
    }
}

/// Command-line mode and configuration parsing for `fluid-helper`.
public enum CLIOptions: Sendable, Equatable {
    case protocolLoop
    case oneShot(wavPath: String, version: ModelVersion, modelsDir: String)
    case diarize(audioPath: String, modelsDir: String, numSpeakers: Int?)
    case help

    public static let defaultModelsRelativePath = "Library/Application Support/com.meetily.ai/models/fluidaudio"
    public static let defaultIdleTimeoutSeconds: TimeInterval = 300.0

    public static func defaultModelsDirectory(
        homeDirectory: String = FileManager.default.homeDirectoryForCurrentUser.path
    ) -> String {
        URL(fileURLWithPath: homeDirectory, isDirectory: true)
            .appendingPathComponent(defaultModelsRelativePath, isDirectory: true)
            .path
    }

    public static func resolveIdleTimeoutSeconds(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> TimeInterval {
        guard
            let raw = environment["FLUID_HELPER_IDLE_SECS"],
            let seconds = Double(raw.trimmingCharacters(in: .whitespacesAndNewlines)),
            seconds > 0
        else {
            return defaultIdleTimeoutSeconds
        }
        return seconds
    }

    /// Parse command-line arguments (excluding `argv[0]`).
    public static func parse(
        arguments: [String],
        defaultModelsDir: String? = nil
    ) throws -> CLIOptions {
        guard !arguments.isEmpty else {
            return .protocolLoop
        }

        if arguments.contains("--help") || arguments.contains("-h") {
            return .help
        }

        if arguments.first == "diarize" {
            return try parseDiarize(arguments: Array(arguments.dropFirst()))
        }

        var wavPath: String?
        var version: ModelVersion = .v3
        var modelsDir: String = defaultModelsDir ?? defaultModelsDirectory()

        var index = 0
        while index < arguments.count {
            let arg = arguments[index]
            switch arg {
            case "--transcribe":
                guard index + 1 < arguments.count else {
                    throw ProtocolError.missingField("--transcribe")
                }
                let candidate = arguments[index + 1].trimmingCharacters(in: .whitespacesAndNewlines)
                guard !candidate.isEmpty, !candidate.hasPrefix("--") else {
                    throw ProtocolError.invalidField(field: "--transcribe", reason: "requires a WAV file path")
                }
                wavPath = (candidate as NSString).expandingTildeInPath
                index += 2

            case "--version":
                guard index + 1 < arguments.count else {
                    throw ProtocolError.missingField("--version")
                }
                let rawVersion = arguments[index + 1]
                guard let parsedVersion = ModelVersion(rawValue: rawVersion) else {
                    throw ProtocolError.unknownVersion(rawVersion)
                }
                version = parsedVersion
                index += 2

            case "--models-dir":
                guard index + 1 < arguments.count else {
                    throw ProtocolError.missingField("--models-dir")
                }
                let rawDir = arguments[index + 1].trimmingCharacters(in: .whitespacesAndNewlines)
                guard !rawDir.isEmpty, !rawDir.hasPrefix("--") else {
                    throw ProtocolError.invalidField(field: "--models-dir", reason: "requires a directory path")
                }
                modelsDir = (rawDir as NSString).expandingTildeInPath
                index += 2

            default:
                throw ProtocolError.invalidField(field: arg, reason: "unknown command-line argument")
            }
        }

        guard let resolvedWavPath = wavPath else {
            throw ProtocolError.missingField("--transcribe")
        }

        return .oneShot(wavPath: resolvedWavPath, version: version, modelsDir: modelsDir)
    }

    private static func parseDiarize(arguments: [String]) throws -> CLIOptions {
        var audioPath: String?
        var modelsDir: String?
        var numSpeakers: Int?

        var index = 0
        while index < arguments.count {
            let arg = arguments[index]
            switch arg {
            case "--audio":
                guard index + 1 < arguments.count else {
                    throw ProtocolError.missingField("--audio")
                }
                let candidate = arguments[index + 1].trimmingCharacters(in: .whitespacesAndNewlines)
                guard !candidate.isEmpty, !candidate.hasPrefix("--") else {
                    throw ProtocolError.invalidField(field: "--audio", reason: "requires a WAV file path")
                }
                audioPath = (candidate as NSString).expandingTildeInPath
                index += 2

            case "--models-dir":
                guard index + 1 < arguments.count else {
                    throw ProtocolError.missingField("--models-dir")
                }
                let rawDir = arguments[index + 1].trimmingCharacters(in: .whitespacesAndNewlines)
                guard !rawDir.isEmpty, !rawDir.hasPrefix("--") else {
                    throw ProtocolError.invalidField(field: "--models-dir", reason: "requires a directory path")
                }
                modelsDir = (rawDir as NSString).expandingTildeInPath
                index += 2

            case "--num-speakers":
                guard index + 1 < arguments.count else {
                    throw ProtocolError.missingField("--num-speakers")
                }
                let rawNum = arguments[index + 1].trimmingCharacters(in: .whitespacesAndNewlines)
                guard let parsed = Int(rawNum), parsed > 0 else {
                    throw ProtocolError.invalidField(
                        field: "--num-speakers",
                        reason: "must be a positive integer"
                    )
                }
                numSpeakers = parsed
                index += 2

            default:
                throw ProtocolError.invalidField(field: arg, reason: "unknown command-line argument")
            }
        }

        guard let resolvedAudioPath = audioPath else {
            throw ProtocolError.missingField("--audio")
        }
        guard let resolvedModelsDir = modelsDir else {
            throw ProtocolError.missingField("--models-dir")
        }

        return .diarize(
            audioPath: resolvedAudioPath,
            modelsDir: resolvedModelsDir,
            numSpeakers: numSpeakers
        )
    }
}
