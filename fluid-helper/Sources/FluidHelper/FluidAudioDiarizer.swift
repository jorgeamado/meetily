import FluidAudio
import FluidHelperCore
import Foundation

/// Runs FluidAudio's offline speaker diarizer (`OfflineDiarizerModels` + `OfflineDiarizerManager`)
/// and converts the resulting segments and 256-d speaker embeddings into `DiarizeOutput`.
enum FluidAudioDiarizer {
    static func diarize(
        audioPath: String,
        modelsDir: String,
        numSpeakers: Int?,
        onProgressPercent: @escaping @Sendable (Int) -> Void
    ) async throws -> DiarizeOutput {
        let expandedAudioPath = (audioPath as NSString).expandingTildeInPath
        let audioURL = URL(fileURLWithPath: expandedAudioPath)
        guard FileManager.default.fileExists(atPath: audioURL.path) else {
            throw ProtocolError.fileNotFound(audioURL.path)
        }

        let expandedModelsDir = (modelsDir as NSString).expandingTildeInPath
        let modelsDirURL = URL(fileURLWithPath: expandedModelsDir, isDirectory: true).standardizedFileURL
        try FileManager.default.createDirectory(
            at: modelsDirURL,
            withIntermediateDirectories: true
        )

        var config = OfflineDiarizerConfig.default
        if let numSpeakers {
            config = config.withSpeakers(exactly: numSpeakers)
        }

        let manager = OfflineDiarizerManager(config: config)

        let progressHandler: ProgressHandler = { progress in
            switch progress.phase {
            case .listing:
                onProgressPercent(2)
            case .downloading(_, let totalFiles):
                guard totalFiles > 0 else { return }
                let scaled = 2 + Int((progress.fractionCompleted * 20.0).rounded())
                onProgressPercent(min(22, max(2, scaled)))
            case .compiling:
                onProgressPercent(25)
            }
        }

        do {
            // Note: Unlike AsrModels.load(from:) which calls deletingLastPathComponent(),
            // OfflineDiarizerModels.load(from:) passes `directory` directly to ModelHub.loadModels(.diarizer, ...),
            // which stores the repository at `<modelsDir>/speaker-diarization`.
            let models = try await OfflineDiarizerModels.load(
                from: modelsDirURL,
                progressHandler: progressHandler
            )
            manager.initialize(models: models)
        } catch let error as URLError {
            // Network failure: keep FluidAudio's partial downloads so the next run resumes.
            throw error
        } catch let error as DownloadError {
            // Transient download failures (stalled, rate limited, HTTP errors) resume on the
            // next run; only a corrupt artifact warrants wiping the cache.
            guard case .invalidArtifact = error else { throw error }
            try await manager.prepareModels(
                directory: modelsDirURL,
                forceRedownload: true
            )
        } catch {
            // Fall back to OfflineDiarizerManager.prepareModels with forceRedownload to recover from a partial/corrupt cache.
            try await manager.prepareModels(
                directory: modelsDirURL,
                forceRedownload: true
            )
        }

        onProgressPercent(30)

        let rawSegments: [RawSpeakerSegment]
        do {
            let result = try await manager.process(audioURL) { chunksProcessed, totalChunks in
                guard totalChunks > 0 else { return }
                let fraction = Double(chunksProcessed) / Double(totalChunks)
                let pct = 30 + Int((fraction * 65.0).rounded())
                onProgressPercent(min(95, max(30, pct)))
            }
            rawSegments = result.segments.map { segment in
                RawSpeakerSegment(
                    speakerId: segment.speakerId,
                    startTimeSeconds: segment.startTimeSeconds,
                    endTimeSeconds: segment.endTimeSeconds,
                    embedding: segment.embedding
                )
            }
        } catch OfflineDiarizationError.noSpeechDetected {
            rawSegments = []
        }

        return DiarizationPostProcessor.buildOutput(from: rawSegments)
    }
}
