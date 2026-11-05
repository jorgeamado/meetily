import Foundation

/// Supported Parakeet TDT model versions exposed by `fluid-helper`.
public enum ModelVersion: String, Codable, Sendable, Equatable, CaseIterable {
    case v3
    case ultra
}

/// Progress stages emitted during a `load` request.
public enum ProgressStage: String, Codable, Sendable, Equatable, CaseIterable {
    case download
    case compile
}

/// Word-level timing span in a transcription result.
public struct WordTimingItem: Codable, Sendable, Equatable {
    public let w: String
    public let start: Double
    public let end: Double

    public init(w: String, start: Double, end: Double) {
        self.w = w
        self.start = start
        self.end = end
    }
}

/// Token-level timing and confidence span in a transcription result.
public struct TokenTimingItem: Codable, Sendable, Equatable {
    public let t: String
    public let start: Double
    public let end: Double
    public let conf: Double

    public init(t: String, start: Double, end: Double, conf: Double) {
        self.t = t
        self.start = start
        self.end = end
        self.conf = conf
    }
}

/// Payload for a `"result"` response.
public struct ResultPayload: Sendable, Equatable {
    public let id: Int
    public let text: String
    public let confidence: Double
    public let duration: Double
    public let processingSecs: Double
    public let words: [WordTimingItem]
    public let tokens: [TokenTimingItem]?

    public init(
        id: Int,
        text: String,
        confidence: Double,
        duration: Double,
        processingSecs: Double,
        words: [WordTimingItem],
        tokens: [TokenTimingItem]? = nil
    ) {
        self.id = id
        self.text = text
        self.confidence = confidence
        self.duration = duration
        self.processingSecs = processingSecs
        self.words = words
        self.tokens = tokens
    }
}

/// Errors raised while decoding requests or handling protocol operations.
public enum ProtocolError: Error, LocalizedError, Sendable, Equatable {
    case malformedJSON(String)
    case missingField(String)
    case invalidField(field: String, reason: String)
    case unknownRequestType(String)
    case unknownVersion(String)
    case modelNotLoaded
    case fileNotFound(String)
    case transcriptionFailed(String)

    public var errorDescription: String? {
        switch self {
        case .malformedJSON(let details):
            return "Malformed JSON: \(details)"
        case .missingField(let field):
            return "Missing required field '\(field)'"
        case .invalidField(let field, let reason):
            return "Invalid field '\(field)': \(reason)"
        case .unknownRequestType(let type):
            return "Unknown request type '\(type)'"
        case .unknownVersion(let version):
            return "Unknown model version '\(version)'. Expected 'v3' or 'ultra'"
        case .modelNotLoaded:
            return "No model loaded. Send a 'load' request before 'transcribe'"
        case .fileNotFound(let path):
            return "Audio file not found at '\(path)'"
        case .transcriptionFailed(let reason):
            return "Transcription failed: \(reason)"
        }
    }
}

/// Inbound protocol request read from a single JSON line on `stdin`.
public enum Request: Sendable, Equatable, Codable {
    case load(version: ModelVersion, modelsDir: String)
    case status(version: ModelVersion, modelsDir: String)
    case transcribe(id: Int, wavPath: String, tokens: Bool)
    case ping
    case shutdown

    private enum CodingKeys: String, CodingKey {
        case type
        case version
        case modelsDir = "models_dir"
        case id
        case wavPath = "wav_path"
        case tokens
    }

    public init(from decoder: Decoder) throws {
        let container: KeyedDecodingContainer<CodingKeys>
        do {
            container = try decoder.container(keyedBy: CodingKeys.self)
        } catch {
            throw ProtocolError.malformedJSON("Expected a JSON object")
        }

        let requestType = try Self.decodeRequired(String.self, forKey: .type, in: container)

        switch requestType {
        case "load":
            let rawVersion = try Self.decodeRequired(String.self, forKey: .version, in: container)
            guard let version = ModelVersion(rawValue: rawVersion) else {
                throw ProtocolError.unknownVersion(rawVersion)
            }
            let modelsDir = try Self.decodeRequired(String.self, forKey: .modelsDir, in: container)
            guard !modelsDir.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw ProtocolError.invalidField(field: "models_dir", reason: "must not be empty")
            }
            self = .load(version: version, modelsDir: modelsDir)

        case "status":
            let rawVersion = try Self.decodeRequired(String.self, forKey: .version, in: container)
            guard let version = ModelVersion(rawValue: rawVersion) else {
                throw ProtocolError.unknownVersion(rawVersion)
            }
            let modelsDir = try Self.decodeRequired(String.self, forKey: .modelsDir, in: container)
            guard !modelsDir.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw ProtocolError.invalidField(field: "models_dir", reason: "must not be empty")
            }
            self = .status(version: version, modelsDir: modelsDir)

        case "transcribe":
            let id = try Self.decodeRequired(Int.self, forKey: .id, in: container)
            let wavPath = try Self.decodeRequired(String.self, forKey: .wavPath, in: container)
            guard !wavPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw ProtocolError.invalidField(field: "wav_path", reason: "must not be empty")
            }
            let includeTokens = try Self.decodeOptionalBool(forKey: .tokens, defaultValue: true, in: container)
            self = .transcribe(id: id, wavPath: wavPath, tokens: includeTokens)

        case "ping":
            self = .ping

        case "shutdown":
            self = .shutdown

        default:
            throw ProtocolError.unknownRequestType(requestType)
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .load(let version, let modelsDir):
            try container.encode("load", forKey: .type)
            try container.encode(version, forKey: .version)
            try container.encode(modelsDir, forKey: .modelsDir)
        case .status(let version, let modelsDir):
            try container.encode("status", forKey: .type)
            try container.encode(version, forKey: .version)
            try container.encode(modelsDir, forKey: .modelsDir)
        case .transcribe(let id, let wavPath, let tokens):
            try container.encode("transcribe", forKey: .type)
            try container.encode(id, forKey: .id)
            try container.encode(wavPath, forKey: .wavPath)
            try container.encode(tokens, forKey: .tokens)
        case .ping:
            try container.encode("ping", forKey: .type)
        case .shutdown:
            try container.encode("shutdown", forKey: .type)
        }
    }

    private static func decodeOptionalBool(
        forKey key: CodingKeys,
        defaultValue: Bool,
        in container: KeyedDecodingContainer<CodingKeys>
    ) throws -> Bool {
        guard container.contains(key) else {
            return defaultValue
        }
        if (try? container.decodeNil(forKey: key)) == true {
            return defaultValue
        }
        return try decodeRequired(Bool.self, forKey: key, in: container)
    }

    private static func decodeRequired<T: Decodable>(
        _ type: T.Type,
        forKey key: CodingKeys,
        in container: KeyedDecodingContainer<CodingKeys>
    ) throws -> T {
        guard container.contains(key) else {
            throw ProtocolError.missingField(key.rawValue)
        }
        if (try? container.decodeNil(forKey: key)) == true {
            throw ProtocolError.missingField(key.rawValue)
        }
        do {
            return try container.decode(T.self, forKey: key)
        } catch let error as ProtocolError {
            throw error
        } catch {
            throw ProtocolError.invalidField(field: key.rawValue, reason: "unexpected value type")
        }
    }
}

/// Outbound protocol response written as a single JSON line to `stdout`.
public enum Response: Sendable, Equatable, Codable {
    case progress(stage: ProgressStage, fraction: Double)
    case loaded(version: ModelVersion, loadSecs: Double)
    case status(version: ModelVersion, downloaded: Bool, path: String)
    case result(ResultPayload)
    case pong
    case goodbye
    case error(id: Int?, message: String)

    private enum CodingKeys: String, CodingKey {
        case type
        case stage
        case fraction
        case version
        case loadSecs = "load_secs"
        case downloaded
        case path
        case id
        case text
        case confidence
        case duration
        case processingSecs = "processing_secs"
        case words
        case tokens
        case message
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .progress(let stage, let fraction):
            try container.encode("progress", forKey: .type)
            try container.encode(stage, forKey: .stage)
            try container.encode(fraction, forKey: .fraction)

        case .loaded(let version, let loadSecs):
            try container.encode("loaded", forKey: .type)
            try container.encode(version, forKey: .version)
            try container.encode(loadSecs, forKey: .loadSecs)

        case .status(let version, let downloaded, let path):
            try container.encode("status", forKey: .type)
            try container.encode(version, forKey: .version)
            try container.encode(downloaded, forKey: .downloaded)
            try container.encode(path, forKey: .path)

        case .result(let payload):
            try container.encode("result", forKey: .type)
            try container.encode(payload.id, forKey: .id)
            try container.encode(payload.text, forKey: .text)
            try container.encode(payload.confidence, forKey: .confidence)
            try container.encode(payload.duration, forKey: .duration)
            try container.encode(payload.processingSecs, forKey: .processingSecs)
            try container.encode(payload.words, forKey: .words)
            try container.encodeIfPresent(payload.tokens, forKey: .tokens)

        case .pong:
            try container.encode("pong", forKey: .type)

        case .goodbye:
            try container.encode("goodbye", forKey: .type)

        case .error(let id, let message):
            try container.encode("error", forKey: .type)
            if let id {
                try container.encode(id, forKey: .id)
            } else {
                try container.encodeNil(forKey: .id)
            }
            try container.encode(message, forKey: .message)
        }
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let responseType = try container.decode(String.self, forKey: .type)

        switch responseType {
        case "progress":
            let stage = try container.decode(ProgressStage.self, forKey: .stage)
            let fraction = try container.decode(Double.self, forKey: .fraction)
            self = .progress(stage: stage, fraction: fraction)

        case "loaded":
            let version = try container.decode(ModelVersion.self, forKey: .version)
            let loadSecs = try container.decode(Double.self, forKey: .loadSecs)
            self = .loaded(version: version, loadSecs: loadSecs)

        case "status":
            let version = try container.decode(ModelVersion.self, forKey: .version)
            let downloaded = try container.decode(Bool.self, forKey: .downloaded)
            let path = try container.decode(String.self, forKey: .path)
            self = .status(version: version, downloaded: downloaded, path: path)

        case "result":
            let id = try container.decode(Int.self, forKey: .id)
            let text = try container.decode(String.self, forKey: .text)
            let confidence = try container.decode(Double.self, forKey: .confidence)
            let duration = try container.decode(Double.self, forKey: .duration)
            let processingSecs = try container.decode(Double.self, forKey: .processingSecs)
            let words = try container.decode([WordTimingItem].self, forKey: .words)
            let tokens = try container.decodeIfPresent([TokenTimingItem].self, forKey: .tokens)
            self = .result(
                ResultPayload(
                    id: id,
                    text: text,
                    confidence: confidence,
                    duration: duration,
                    processingSecs: processingSecs,
                    words: words,
                    tokens: tokens
                )
            )

        case "pong":
            self = .pong

        case "goodbye":
            self = .goodbye

        case "error":
            let id = try container.decodeIfPresent(Int.self, forKey: .id)
            let message = try container.decode(String.self, forKey: .message)
            self = .error(id: id, message: message)

        default:
            throw ProtocolError.unknownRequestType(responseType)
        }
    }
}

/// JSON line encoding and decoding utilities for `Request` and `Response`.
public enum ProtocolCodec {
    /// Decode a single UTF-8 JSON line into a `Request`.
    public static func decodeRequest(from line: String) throws -> Request {
        let data = Data(line.utf8)
        do {
            return try JSONDecoder().decode(Request.self, from: data)
        } catch let error as ProtocolError {
            throw error
        } catch let error as DecodingError {
            throw ProtocolError.malformedJSON(describeDecodingError(error))
        } catch {
            throw ProtocolError.malformedJSON(error.localizedDescription)
        }
    }

    /// Encode a `Response` as a compact single-line UTF-8 JSON string (without trailing newline).
    public static func encodeResponse(_ response: Response) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.withoutEscapingSlashes]
        let data = try encoder.encode(response)
        guard let line = String(data: data, encoding: .utf8) else {
            throw ProtocolError.malformedJSON("Failed to encode response as UTF-8")
        }
        return line
    }

    /// Encode a `DiarizeOutput` as a compact single-line UTF-8 JSON string (without trailing newline).
    public static func encodeDiarizeOutput(_ output: DiarizeOutput) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.withoutEscapingSlashes]
        let data = try encoder.encode(output)
        guard let line = String(data: data, encoding: .utf8) else {
            throw ProtocolError.malformedJSON("Failed to encode diarize output as UTF-8")
        }
        return line
    }

    /// Format any `Error` into a human-readable message suitable for `Response.error`.
    public static func describeError(_ error: Error) -> String {
        if let protocolError = error as? ProtocolError, let description = protocolError.errorDescription {
            return description
        }
        if let decodingError = error as? DecodingError {
            return describeDecodingError(decodingError)
        }
        if let localized = error as? LocalizedError, let description = localized.errorDescription {
            return description
        }
        return error.localizedDescription
    }

    private static func describeDecodingError(_ error: DecodingError) -> String {
        switch error {
        case .keyNotFound(let key, _):
            return "Missing required field '\(key.stringValue)'"
        case .valueNotFound(_, let context):
            let key = context.codingPath.last?.stringValue ?? "root"
            return "Missing value for field '\(key)'"
        case .typeMismatch(_, let context):
            let key = context.codingPath.last?.stringValue ?? "root"
            return "Type mismatch at '\(key)': \(context.debugDescription)"
        case .dataCorrupted(let context):
            return context.debugDescription
        @unknown default:
            return error.localizedDescription
        }
    }
}

// MARK: - One-Shot Diarization Output Types

/// Raw speaker turn produced by FluidAudio's offline diarizer prior to index mapping and centroid computation.
public struct RawSpeakerSegment: Sendable, Equatable {
    public let speakerId: String
    public let startTimeSeconds: Float
    public let endTimeSeconds: Float
    public let embedding: [Float]

    public init(
        speakerId: String,
        startTimeSeconds: Float,
        endTimeSeconds: Float,
        embedding: [Float]
    ) {
        self.speakerId = speakerId
        self.startTimeSeconds = startTimeSeconds
        self.endTimeSeconds = endTimeSeconds
        self.embedding = embedding
    }
}

/// Diarized speech segment in the `diarize-helper` stdout JSON format.
public struct DiarizeSegmentItem: Codable, Sendable, Equatable {
    public let start: Float
    public let end: Float
    public let speaker: Int
    public let voice: Bool

    public init(start: Float, end: Float, speaker: Int, voice: Bool = false) {
        self.start = start
        self.end = end
        self.speaker = speaker
        self.voice = voice
    }
}

/// Per-speaker L2-normalised centroid in the `diarize-helper` stdout JSON format.
public struct DiarizeCentroidItem: Codable, Sendable, Equatable {
    public let speaker: Int
    public let cleanSecs: Float
    public let embedding: [Float]

    private enum CodingKeys: String, CodingKey {
        case speaker
        case cleanSecs = "clean_secs"
        case embedding
    }

    public init(speaker: Int, cleanSecs: Float, embedding: [Float]) {
        self.speaker = speaker
        self.cleanSecs = cleanSecs
        self.embedding = embedding
    }
}

/// Top-level stdout JSON object emitted by `fluid-helper diarize`.
public struct DiarizeOutput: Codable, Sendable, Equatable {
    public let segments: [DiarizeSegmentItem]
    public let numSpeakers: Int
    public let centroids: [DiarizeCentroidItem]

    private enum CodingKeys: String, CodingKey {
        case segments
        case numSpeakers = "num_speakers"
        case centroids
    }

    public init(
        segments: [DiarizeSegmentItem],
        numSpeakers: Int,
        centroids: [DiarizeCentroidItem]
    ) {
        self.segments = segments
        self.numSpeakers = numSpeakers
        self.centroids = centroids
    }
}

/// Pure post-processing for converting raw FluidAudio segments into the `diarize-helper` output schema.
public enum DiarizationPostProcessor {
    /// Sort segments by start time, map `speakerId` strings to 0-based integers in order of first appearance,
    /// and compute duration-weighted L2-normalised centroids per speaker.
    public static func buildOutput(from rawSegments: [RawSpeakerSegment]) -> DiarizeOutput {
        guard !rawSegments.isEmpty else {
            return DiarizeOutput(segments: [], numSpeakers: 0, centroids: [])
        }

        let sortedIndexed = rawSegments.enumerated().sorted { lhs, rhs in
            if lhs.element.startTimeSeconds != rhs.element.startTimeSeconds {
                return lhs.element.startTimeSeconds < rhs.element.startTimeSeconds
            }
            if lhs.element.endTimeSeconds != rhs.element.endTimeSeconds {
                return lhs.element.endTimeSeconds < rhs.element.endTimeSeconds
            }
            return lhs.offset < rhs.offset
        }

        var speakerIndexById: [String: Int] = [:]
        var nextSpeakerIndex = 0
        var mappedSegments: [DiarizeSegmentItem] = []
        mappedSegments.reserveCapacity(sortedIndexed.count)

        var segmentsBySpeaker: [[RawSpeakerSegment]] = []

        for (_, raw) in sortedIndexed {
            let speakerIdx: Int
            if let existing = speakerIndexById[raw.speakerId] {
                speakerIdx = existing
            } else {
                speakerIdx = nextSpeakerIndex
                speakerIndexById[raw.speakerId] = speakerIdx
                nextSpeakerIndex += 1
                segmentsBySpeaker.append([])
            }

            mappedSegments.append(
                DiarizeSegmentItem(
                    start: raw.startTimeSeconds,
                    end: raw.endTimeSeconds,
                    speaker: speakerIdx,
                    voice: false
                )
            )
            segmentsBySpeaker[speakerIdx].append(raw)
        }

        var centroids: [DiarizeCentroidItem] = []
        centroids.reserveCapacity(nextSpeakerIndex)

        for speakerIdx in 0..<nextSpeakerIndex {
            let speakerSegments = segmentsBySpeaker[speakerIdx]
            var cleanSecs: Float = 0.0
            var weightedSum: [Float] = []
            var weightedDuration: Float = 0.0

            for segment in speakerSegments {
                let duration = max(0.0, segment.endTimeSeconds - segment.startTimeSeconds)
                cleanSecs += duration

                guard duration > 0.0, !segment.embedding.isEmpty, segment.embedding.allSatisfy(\.isFinite) else {
                    continue
                }

                if weightedSum.isEmpty {
                    weightedSum = segment.embedding.map { $0 * duration }
                    weightedDuration = duration
                } else if segment.embedding.count == weightedSum.count {
                    for index in 0..<weightedSum.count {
                        weightedSum[index] += segment.embedding[index] * duration
                    }
                    weightedDuration += duration
                }
            }

            guard !weightedSum.isEmpty, weightedDuration > 0.0 else {
                continue
            }

            let mean = weightedSum.map { $0 / weightedDuration }
            let squaredNorm = mean.reduce(Float(0.0)) { $0 + $1 * $1 }
            let norm = squaredNorm.squareRoot()
            guard norm > 1e-6 else {
                continue
            }

            let normalized = mean.map { $0 / norm }
            centroids.append(
                DiarizeCentroidItem(
                    speaker: speakerIdx,
                    cleanSecs: cleanSecs,
                    embedding: normalized
                )
            )
        }

        return DiarizeOutput(
            segments: mappedSegments,
            numSpeakers: nextSpeakerIndex,
            centroids: centroids
        )
    }
}
