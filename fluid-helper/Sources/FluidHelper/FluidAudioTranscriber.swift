import FluidAudio
import FluidHelperCore
import Foundation

extension ModelVersion {
    var fluidAudioVersion: AsrModelVersion {
        switch self {
        case .v3:
            return .v3
        case .ultra:
            return .ultra
        }
    }

    var fluidAudioRepo: Repo {
        switch self {
        case .v3:
            return .parakeetV3
        case .ultra:
            return .parakeetUltra
        }
    }
}

extension LoadProgressBridge {
    func handle(_ progress: DownloadProgress) {
        switch progress.phase {
        case .listing:
            handleListing()
        case .downloading(_, let totalFiles):
            handleDownloading(fractionCompleted: progress.fractionCompleted, totalFiles: totalFiles)
        case .compiling(let modelName):
            handleCompiling(modelName: modelName)
        }
    }
}

/// Real `Transcriber` implementation backed by FluidAudio v0.18.2 `AsrModels` and `AsrManager`.
actor FluidAudioTranscriber: Transcriber {
    private var asrManager: AsrManager?
    private var loadedVersion: ModelVersion?

    init() {}

    private static func resolveModelDirectory(version: ModelVersion, modelsDir: String) -> URL {
        let expandedBaseDir = (modelsDir as NSString).expandingTildeInPath
        let baseDirURL = URL(fileURLWithPath: expandedBaseDir, isDirectory: true)
        return baseDirURL.appendingPathComponent(
            version.fluidAudioRepo.folderName,
            isDirectory: true
        )
    }

    func modelStatus(
        version: ModelVersion,
        modelsDir: String
    ) async throws -> ModelStatusOutput {
        let modelDirURL = Self.resolveModelDirectory(version: version, modelsDir: modelsDir)
        let downloaded = AsrModels.modelsExist(
            at: modelDirURL,
            version: version.fluidAudioVersion
        )
        return ModelStatusOutput(
            downloaded: downloaded,
            path: modelDirURL.path
        )
    }

    func load(
        version: ModelVersion,
        modelsDir: String,
        onProgress: @escaping @Sendable (ProgressStage, Double) -> Void
    ) async throws -> Double {
        if loadedVersion == version, asrManager != nil {
            return 0.0
        }

        if let existingManager = asrManager {
            await existingManager.cleanup()
            asrManager = nil
            loadedVersion = nil
        }

        let startTime = Date()
        let modelDirURL = Self.resolveModelDirectory(version: version, modelsDir: modelsDir)

        let asrVersion = version.fluidAudioVersion
        let bridge = LoadProgressBridge(onProgress: onProgress)
        let progressHandler: ProgressHandler = { progress in
            bridge.handle(progress)
        }

        let models = try await AsrModels.downloadAndLoad(
            to: modelDirURL,
            version: asrVersion,
            progressHandler: progressHandler
        )

        let tdtConfig = TdtConfig(blankId: asrVersion.blankId)
        let asrConfig = ASRConfig(
            tdtConfig: tdtConfig,
            encoderHiddenSize: asrVersion.encoderHiddenSize
        )
        let manager = AsrManager(config: asrConfig)
        try await manager.loadModels(models)
        bridge.complete()

        asrManager = manager
        loadedVersion = version

        return Date().timeIntervalSince(startTime)
    }

    func transcribe(
        wavPath: String,
        includeTokens: Bool
    ) async throws -> TranscriptionOutput {
        guard let manager = asrManager else {
            throw ProtocolError.modelNotLoaded
        }

        let expandedWavPath = (wavPath as NSString).expandingTildeInPath
        let wavURL = URL(fileURLWithPath: expandedWavPath)
        guard FileManager.default.fileExists(atPath: wavURL.path) else {
            throw ProtocolError.fileNotFound(wavURL.path)
        }

        let decoderLayers = await manager.decoderLayerCount
        var decoderState = TdtDecoderState.make(decoderLayers: decoderLayers)
        let result = try await manager.transcribe(wavURL, decoderState: &decoderState)

        let rawTokenTimings = result.tokenTimings ?? []
        let fluidWords = buildWordTimings(from: rawTokenTimings)
        let words = fluidWords.map { wordTiming in
            WordTimingItem(
                w: wordTiming.word,
                start: wordTiming.startTime,
                end: wordTiming.endTime
            )
        }
        let tokens: [TokenTimingItem] = includeTokens
            ? rawTokenTimings.map { timing in
                TokenTimingItem(
                    t: timing.token,
                    start: timing.startTime,
                    end: timing.endTime,
                    conf: Double(timing.confidence)
                )
            }
            : []

        return TranscriptionOutput(
            text: result.text,
            confidence: Double(result.confidence),
            duration: result.duration,
            processingSecs: result.processingTime,
            words: words,
            tokens: tokens
        )
    }
}
