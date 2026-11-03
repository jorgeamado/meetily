import Darwin
import FluidAudio
import FluidHelperCore
import Foundation
import os

/// Thread-safe writer that serializes JSON responses to a dedicated file descriptor duplicated from the original
/// `STDOUT_FILENO`, while `STDOUT_FILENO` (fd 1) itself is redirected to `STDERR_FILENO` (fd 2).
private final class StdoutResponseWriter: Sendable {
    private let fileDescriptor: Int32
    private let lock = OSAllocatedUnfairLock()

    init(fileDescriptor: Int32) {
        self.fileDescriptor = fileDescriptor
    }

    /// Duplicate `STDOUT_FILENO` for the JSON response channel and redirect `STDOUT_FILENO` to `STDERR_FILENO`
    /// so any `print()`, `FileHandle.standardOutput`, or C `stdout` writes go to `stderr`.
    static func installRedirectingStdoutToStderr() -> StdoutResponseWriter {
        fflush(stdout)
        let savedFD = Darwin.dup(STDOUT_FILENO)
        guard savedFD >= 0 else {
            StderrLogger.log("Fatal: dup(STDOUT_FILENO) failed with errno \(errno)")
            Darwin.exit(1)
        }
        _ = Darwin.fcntl(savedFD, F_SETFD, FD_CLOEXEC)

        guard Darwin.dup2(STDERR_FILENO, STDOUT_FILENO) >= 0 else {
            StderrLogger.log("Fatal: dup2(STDERR_FILENO, STDOUT_FILENO) failed with errno \(errno)")
            Darwin.close(savedFD)
            Darwin.exit(1)
        }

        return StdoutResponseWriter(fileDescriptor: savedFD)
    }

    func write(_ response: Response) {
        guard let jsonLine = try? ProtocolCodec.encodeResponse(response) else {
            StderrLogger.log("Failed to encode response")
            return
        }
        let payload = Data((jsonLine + "\n").utf8)
        let fd = fileDescriptor
        lock.withLock {
            payload.withUnsafeBytes { rawBuffer in
                guard let baseAddress = rawBuffer.baseAddress else { return }
                var offset = 0
                while offset < rawBuffer.count {
                    let written = Darwin.write(
                        fd,
                        baseAddress.advanced(by: offset),
                        rawBuffer.count - offset
                    )
                    if written > 0 {
                        offset += written
                    } else if written < 0 && errno == EINTR {
                        continue
                    } else {
                        break
                    }
                }
            }
        }
    }
}

/// Thread-safe logger that writes exclusively to `stderr`.
private enum StderrLogger {
    private static let lock = OSAllocatedUnfairLock()

    static func log(_ message: String) {
        let payload = Data(("[fluid-helper] " + message + "\n").utf8)
        lock.withLock {
            try? FileHandle.standardError.write(contentsOf: payload)
        }
    }
}

@main
struct FluidHelperMain {
    static func main() async {
        // Duplicate original stdout for JSON responses and redirect fd 1 -> fd 2 before any FluidAudio call.
        let writer = StdoutResponseWriter.installRedirectingStdoutToStderr()

        // Ensure FluidAudio console logs only go to stderr.
        AppLogger.mirrorsToConsole = true

        let arguments = Array(CommandLine.arguments.dropFirst())
        let options: CLIOptions
        do {
            options = try CLIOptions.parse(arguments: arguments)
        } catch {
            StderrLogger.log("Error: \(ProtocolCodec.describeError(error))")
            printUsageToStderr()
            Darwin.exit(1)
        }

        switch options {
        case .help:
            printUsageToStderr()
            Darwin.exit(0)

        case .oneShot(let wavPath, let version, let modelsDir):
            await runOneShot(
                wavPath: wavPath,
                version: version,
                modelsDir: modelsDir,
                writer: writer
            )

        case .protocolLoop:
            await runProtocolLoop(writer: writer)
        }
    }

    private static func runOneShot(
        wavPath: String,
        version: ModelVersion,
        modelsDir: String,
        writer: StdoutResponseWriter
    ) async {
        let transcriber = FluidAudioTranscriber()

        do {
            let loadSecs = try await transcriber.load(
                version: version,
                modelsDir: modelsDir,
                onProgress: { stage, fraction in
                    let pct = Int((fraction * 100.0).rounded())
                    StderrLogger.log("Progress [\(stage.rawValue)]: \(pct)%")
                }
            )
            StderrLogger.log(
                "Loaded model '\(version.rawValue)' in \(String(format: "%.2f", loadSecs))s"
            )

            let output = try await transcriber.transcribe(wavPath: wavPath, includeTokens: true)
            let resultPayload = ResultPayload(
                id: 0,
                text: output.text,
                confidence: output.confidence,
                duration: output.duration,
                processingSecs: output.processingSecs,
                words: output.words,
                tokens: output.tokens
            )
            writer.write(.result(resultPayload))
            Darwin.exit(0)
        } catch {
            StderrLogger.log("One-shot transcription failed: \(ProtocolCodec.describeError(error))")
            Darwin.exit(1)
        }
    }

    private static func runProtocolLoop(writer: StdoutResponseWriter) async {
        let idleTimeoutSecs = CLIOptions.resolveIdleTimeoutSeconds()
        StderrLogger.log("Starting protocol loop (idle timeout: \(Int(idleTimeoutSecs))s)")

        let activityTracker = ActivityTracker()
        let watchdogTask = startIdleWatchdog(
            activityTracker: activityTracker,
            idleTimeoutSecs: idleTimeoutSecs
        )
        defer { watchdogTask.cancel() }

        let session = ProtocolSession(transcriber: FluidAudioTranscriber())

        while true {
            if activityTracker.isIdleExpired(timeoutSeconds: idleTimeoutSecs) {
                StderrLogger.log("Idle timeout reached, shutting down")
                break
            }

            guard let line = Swift.readLine(strippingNewline: true) else {
                StderrLogger.log("EOF received on stdin, shutting down")
                break
            }

            if activityTracker.isIdleExpired(timeoutSeconds: idleTimeoutSecs) {
                StderrLogger.log("Idle timeout reached, shutting down")
                break
            }

            if line.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                continue
            }

            activityTracker.beginRequest()
            let shouldContinue = await session.handleLine(line) { response in
                writer.write(response)
            }
            activityTracker.endRequest()

            guard shouldContinue else {
                StderrLogger.log("Shutdown requested, exiting")
                break
            }
        }

        StderrLogger.log("Exiting")
        Darwin.exit(0)
    }

    private static func startIdleWatchdog(
        activityTracker: ActivityTracker,
        idleTimeoutSecs: TimeInterval
    ) -> Task<Void, Never> {
        let pollIntervalSecs = min(1.0, max(0.05, idleTimeoutSecs / 4.0))
        return Task.detached {
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(pollIntervalSecs))
                guard !Task.isCancelled else { return }
                if activityTracker.isIdleExpired(timeoutSeconds: idleTimeoutSecs) {
                    StderrLogger.log("Idle timeout (\(Int(idleTimeoutSecs))s) reached, shutting down")
                    Darwin.exit(0)
                }
            }
        }
    }

    private static func printUsageToStderr() {
        let usage = """
            Usage:
              fluid-helper
                Run persistent JSON-lines protocol over stdin/stdout.

              fluid-helper --transcribe <wav> [--version v3|ultra] [--models-dir <dir>]
                One-shot mode: load model, transcribe <wav>, print single 'result' JSON line to stdout, and exit.
            """
        StderrLogger.log(usage)
    }
}
