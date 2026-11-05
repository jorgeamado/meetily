import Foundation
import os
import Testing
@testable import FluidHelperCore

/// Fake `Transcriber` implementation used to test `ProtocolSession` end-to-end without FluidAudio/CoreML.
actor FakeTranscriber: Transcriber {
    struct LoadCall: Sendable, Equatable {
        let version: ModelVersion
        let modelsDir: String
    }

    struct StatusCall: Sendable, Equatable {
        let version: ModelVersion
        let modelsDir: String
    }

    struct TranscribeCall: Sendable, Equatable {
        let wavPath: String
        let includeTokens: Bool
    }

    private(set) var loadCalls: [LoadCall] = []
    private(set) var statusCalls: [StatusCall] = []
    private(set) var transcribeCalls: [TranscribeCall] = []
    var simulatedLoadSecs: Double = 1.25
    var downloadedVersions: Set<ModelVersion> = []
    var failingWavPaths: Set<String> = []
    var failingLoadVersions: Set<ModelVersion> = []
    var remainingLoadFailures: Int = 0

    func setDownloadedVersions(_ versions: Set<ModelVersion>) {
        downloadedVersions = versions
    }

    func setFailingWavPaths(_ paths: Set<String>) {
        failingWavPaths = paths
    }

    func setFailingLoadVersions(_ versions: Set<ModelVersion>) {
        failingLoadVersions = versions
    }

    func setRemainingLoadFailures(_ count: Int) {
        remainingLoadFailures = count
    }

    private static func fakeFolderName(for version: ModelVersion) -> String {
        switch version {
        case .v3:
            return "parakeet-tdt-0.6b-v3"
        case .ultra:
            return "parakeet-ultra"
        }
    }

    func modelStatus(
        version: ModelVersion,
        modelsDir: String
    ) async throws -> ModelStatusOutput {
        statusCalls.append(StatusCall(version: version, modelsDir: modelsDir))
        let expandedBaseDir = (modelsDir as NSString).expandingTildeInPath
        let baseDirURL = URL(fileURLWithPath: expandedBaseDir, isDirectory: true)
        let modelDirURL = baseDirURL.appendingPathComponent(
            Self.fakeFolderName(for: version),
            isDirectory: true
        )
        return ModelStatusOutput(
            downloaded: downloadedVersions.contains(version),
            path: modelDirURL.path
        )
    }

    func load(
        version: ModelVersion,
        modelsDir: String,
        onProgress: @escaping @Sendable (ProgressStage, Double) -> Void
    ) async throws -> Double {
        loadCalls.append(LoadCall(version: version, modelsDir: modelsDir))
        if remainingLoadFailures > 0 {
            remainingLoadFailures -= 1
            throw ProtocolError.transcriptionFailed("Simulated transient load failure for \(version.rawValue)")
        }
        if failingLoadVersions.contains(version) {
            throw ProtocolError.transcriptionFailed("Simulated load failure for \(version.rawValue)")
        }
        onProgress(.download, 0.5)
        onProgress(.compile, 1.0)
        downloadedVersions.insert(version)
        return simulatedLoadSecs
    }

    func transcribe(
        wavPath: String,
        includeTokens: Bool
    ) async throws -> TranscriptionOutput {
        transcribeCalls.append(TranscribeCall(wavPath: wavPath, includeTokens: includeTokens))
        if failingWavPaths.contains(wavPath) {
            throw ProtocolError.fileNotFound(wavPath)
        }
        let words = [
            WordTimingItem(w: "forecast", start: 12.34, end: 12.80)
        ]
        let tokens = [
            TokenTimingItem(t: "▁fore", start: 12.34, end: 12.50, conf: 0.98),
            TokenTimingItem(t: "cast", start: 12.50, end: 12.80, conf: 0.96),
        ]
        return TranscriptionOutput(
            text: "forecast",
            confidence: 0.93,
            duration: 120.1,
            processingSecs: 1.6,
            words: words,
            tokens: includeTokens ? tokens : []
        )
    }
}

@Suite("FluidHelperCore Protocol & Session Tests")
struct FluidHelperCoreTests {

    // MARK: - Request Decoding

    @Test("Decode every valid request type")
    func decodeValidRequests() throws {
        let loadV3 = try ProtocolCodec.decodeRequest(
            from: #"{"type":"load","version":"v3","models_dir":"/abs/path"}"#
        )
        #expect(loadV3 == .load(version: .v3, modelsDir: "/abs/path"))

        let loadUltra = try ProtocolCodec.decodeRequest(
            from: #"{"type":"load","version":"ultra","models_dir":"/abs/models"}"#
        )
        #expect(loadUltra == .load(version: .ultra, modelsDir: "/abs/models"))

        let statusV3 = try ProtocolCodec.decodeRequest(
            from: #"{"type":"status","version":"v3","models_dir":"/abs/path"}"#
        )
        #expect(statusV3 == .status(version: .v3, modelsDir: "/abs/path"))

        let statusUltra = try ProtocolCodec.decodeRequest(
            from: #"{"type":"status","version":"ultra","models_dir":"/abs/models"}"#
        )
        #expect(statusUltra == .status(version: .ultra, modelsDir: "/abs/models"))

        let transcribeDefault = try ProtocolCodec.decodeRequest(
            from: #"{"type":"transcribe","id":7,"wav_path":"/abs/file.wav"}"#
        )
        #expect(transcribeDefault == .transcribe(id: 7, wavPath: "/abs/file.wav", tokens: true))

        let transcribeNoTokens = try ProtocolCodec.decodeRequest(
            from: #"{"type":"transcribe","id":9,"wav_path":"/abs/file.wav","tokens":false}"#
        )
        #expect(transcribeNoTokens == .transcribe(id: 9, wavPath: "/abs/file.wav", tokens: false))

        let transcribeWithTokens = try ProtocolCodec.decodeRequest(
            from: #"{"type":"transcribe","id":10,"wav_path":"/abs/file.wav","tokens":true}"#
        )
        #expect(transcribeWithTokens == .transcribe(id: 10, wavPath: "/abs/file.wav", tokens: true))

        let ping = try ProtocolCodec.decodeRequest(from: #"{"type":"ping"}"#)
        #expect(ping == .ping)

        let shutdown = try ProtocolCodec.decodeRequest(from: #"{"type":"shutdown"}"#)
        #expect(shutdown == .shutdown)
    }

    @Test("Reject unknown request types, unknown versions, missing fields, and malformed JSON")
    func rejectInvalidRequests() {
        #expect(throws: ProtocolError.unknownRequestType("bogus")) {
            _ = try ProtocolCodec.decodeRequest(from: #"{"type":"bogus"}"#)
        }

        #expect(throws: ProtocolError.missingField("type")) {
            _ = try ProtocolCodec.decodeRequest(from: #"{"version":"v3"}"#)
        }

        #expect(throws: ProtocolError.unknownVersion("v2")) {
            _ = try ProtocolCodec.decodeRequest(
                from: #"{"type":"load","version":"v2","models_dir":"/abs/path"}"#
            )
        }

        #expect(throws: ProtocolError.missingField("version")) {
            _ = try ProtocolCodec.decodeRequest(from: #"{"type":"load","models_dir":"/abs/path"}"#)
        }

        #expect(throws: ProtocolError.missingField("models_dir")) {
            _ = try ProtocolCodec.decodeRequest(from: #"{"type":"load","version":"v3"}"#)
        }

        #expect(throws: ProtocolError.invalidField(field: "models_dir", reason: "must not be empty")) {
            _ = try ProtocolCodec.decodeRequest(
                from: #"{"type":"load","version":"v3","models_dir":"   "}"#
            )
        }

        #expect(throws: ProtocolError.unknownVersion("v2")) {
            _ = try ProtocolCodec.decodeRequest(
                from: #"{"type":"status","version":"v2","models_dir":"/abs/path"}"#
            )
        }

        #expect(throws: ProtocolError.missingField("version")) {
            _ = try ProtocolCodec.decodeRequest(from: #"{"type":"status","models_dir":"/abs/path"}"#)
        }

        #expect(throws: ProtocolError.missingField("models_dir")) {
            _ = try ProtocolCodec.decodeRequest(from: #"{"type":"status","version":"v3"}"#)
        }

        #expect(throws: ProtocolError.invalidField(field: "models_dir", reason: "must not be empty")) {
            _ = try ProtocolCodec.decodeRequest(
                from: #"{"type":"status","version":"v3","models_dir":"   "}"#
            )
        }

        #expect(throws: ProtocolError.missingField("id")) {
            _ = try ProtocolCodec.decodeRequest(
                from: #"{"type":"transcribe","wav_path":"/abs/file.wav"}"#
            )
        }

        #expect(throws: ProtocolError.invalidField(field: "id", reason: "unexpected value type")) {
            _ = try ProtocolCodec.decodeRequest(
                from: #"{"type":"transcribe","id":"not-an-int","wav_path":"/abs/file.wav"}"#
            )
        }

        #expect(throws: ProtocolError.missingField("wav_path")) {
            _ = try ProtocolCodec.decodeRequest(from: #"{"type":"transcribe","id":7}"#)
        }

        #expect(throws: ProtocolError.invalidField(field: "wav_path", reason: "must not be empty")) {
            _ = try ProtocolCodec.decodeRequest(
                from: #"{"type":"transcribe","id":7,"wav_path":""}"#
            )
        }

        #expect(throws: ProtocolError.invalidField(field: "tokens", reason: "unexpected value type")) {
            _ = try ProtocolCodec.decodeRequest(
                from: #"{"type":"transcribe","id":7,"wav_path":"/a.wav","tokens":"false"}"#
            )
        }

        #expect(throws: ProtocolError.self) {
            _ = try ProtocolCodec.decodeRequest(from: "not valid json")
        }

        #expect(throws: ProtocolError.self) {
            _ = try ProtocolCodec.decodeRequest(from: "[1, 2, 3]")
        }
    }

    // MARK: - Response Encoding

    @Test("Encode every response type to exact snake_case JSON keys with no extra top-level keys")
    func encodeEveryResponseTypeExactKeys() throws {
        // 1. progress (download & compile)
        let progressDownloadJSON = try ProtocolCodec.encodeResponse(
            .progress(stage: .download, fraction: 0.4)
        )
        let progressDownloadObj = try parseJSONObject(progressDownloadJSON)
        #expect(Set(progressDownloadObj.keys) == Set(["type", "stage", "fraction"]))
        #expect(progressDownloadObj["type"] as? String == "progress")
        #expect(progressDownloadObj["stage"] as? String == "download")
        #expect(progressDownloadObj["fraction"] as? Double == 0.4)

        let progressCompileJSON = try ProtocolCodec.encodeResponse(
            .progress(stage: .compile, fraction: 0.75)
        )
        let progressCompileObj = try parseJSONObject(progressCompileJSON)
        #expect(Set(progressCompileObj.keys) == Set(["type", "stage", "fraction"]))
        #expect(progressCompileObj["type"] as? String == "progress")
        #expect(progressCompileObj["stage"] as? String == "compile")
        #expect(progressCompileObj["fraction"] as? Double == 0.75)

        // 2. loaded
        let loadedJSON = try ProtocolCodec.encodeResponse(.loaded(version: .v3, loadSecs: 12.3))
        let loadedObj = try parseJSONObject(loadedJSON)
        #expect(Set(loadedObj.keys) == Set(["type", "version", "load_secs"]))
        #expect(loadedObj["type"] as? String == "loaded")
        #expect(loadedObj["version"] as? String == "v3")
        #expect(loadedObj["load_secs"] as? Double == 12.3)

        // 2b. status
        let statusJSON = try ProtocolCodec.encodeResponse(
            .status(version: .v3, downloaded: true, path: "/abs/models/parakeet-tdt-0.6b-v3")
        )
        let statusObj = try parseJSONObject(statusJSON)
        #expect(Set(statusObj.keys) == Set(["type", "version", "downloaded", "path"]))
        #expect(statusObj["type"] as? String == "status")
        #expect(statusObj["version"] as? String == "v3")
        #expect(statusObj["downloaded"] as? Bool == true)
        #expect(statusObj["path"] as? String == "/abs/models/parakeet-tdt-0.6b-v3")

        // 3. result (with tokens)
        let resultWithTokens = Response.result(
            ResultPayload(
                id: 7,
                text: "forecast",
                confidence: 0.93,
                duration: 120.1,
                processingSecs: 1.6,
                words: [WordTimingItem(w: "forecast", start: 12.34, end: 12.80)],
                tokens: [TokenTimingItem(t: "▁fore", start: 12.34, end: 12.50, conf: 0.98)]
            )
        )
        let resultJSON = try ProtocolCodec.encodeResponse(resultWithTokens)
        let resultObj = try parseJSONObject(resultJSON)
        #expect(
            Set(resultObj.keys)
                == Set(["type", "id", "text", "confidence", "duration", "processing_secs", "words", "tokens"])
        )
        #expect(resultObj["type"] as? String == "result")
        #expect(resultObj["id"] as? Int == 7)
        #expect(resultObj["text"] as? String == "forecast")
        #expect(resultObj["confidence"] as? Double == 0.93)
        #expect(resultObj["duration"] as? Double == 120.1)
        #expect(resultObj["processing_secs"] as? Double == 1.6)

        let wordsArray = try #require(resultObj["words"] as? [[String: Any]])
        #expect(wordsArray.count == 1)
        #expect(Set(wordsArray[0].keys) == Set(["w", "start", "end"]))
        #expect(wordsArray[0]["w"] as? String == "forecast")
        #expect(wordsArray[0]["start"] as? Double == 12.34)
        #expect(wordsArray[0]["end"] as? Double == 12.80)

        let tokensArray = try #require(resultObj["tokens"] as? [[String: Any]])
        #expect(tokensArray.count == 1)
        #expect(Set(tokensArray[0].keys) == Set(["t", "start", "end", "conf"]))
        #expect(tokensArray[0]["t"] as? String == "▁fore")
        #expect(tokensArray[0]["start"] as? Double == 12.34)
        #expect(tokensArray[0]["end"] as? Double == 12.50)
        #expect(tokensArray[0]["conf"] as? Double == 0.98)

        // 4. result (without tokens, omitted when tokens == nil)
        let resultWithoutTokens = Response.result(
            ResultPayload(
                id: 8,
                text: "forecast",
                confidence: 0.93,
                duration: 120.1,
                processingSecs: 1.6,
                words: [WordTimingItem(w: "forecast", start: 12.34, end: 12.80)],
                tokens: nil
            )
        )
        let resultNoTokensJSON = try ProtocolCodec.encodeResponse(resultWithoutTokens)
        let resultNoTokensObj = try parseJSONObject(resultNoTokensJSON)
        #expect(
            Set(resultNoTokensObj.keys)
                == Set(["type", "id", "text", "confidence", "duration", "processing_secs", "words"])
        )
        #expect(resultNoTokensObj["tokens"] == nil)

        // 5. pong
        let pongJSON = try ProtocolCodec.encodeResponse(.pong)
        let pongObj = try parseJSONObject(pongJSON)
        #expect(Set(pongObj.keys) == Set(["type"]))
        #expect(pongObj["type"] as? String == "pong")

        // 6. goodbye
        let goodbyeJSON = try ProtocolCodec.encodeResponse(.goodbye)
        let goodbyeObj = try parseJSONObject(goodbyeJSON)
        #expect(Set(goodbyeObj.keys) == Set(["type"]))
        #expect(goodbyeObj["type"] as? String == "goodbye")

        // 7. error (with integer id)
        let errorWithIdJSON = try ProtocolCodec.encodeResponse(.error(id: 7, message: "bad wav"))
        let errorWithIdObj = try parseJSONObject(errorWithIdJSON)
        #expect(Set(errorWithIdObj.keys) == Set(["type", "id", "message"]))
        #expect(errorWithIdObj["type"] as? String == "error")
        #expect(errorWithIdObj["id"] as? Int == 7)
        #expect(errorWithIdObj["message"] as? String == "bad wav")

        // 8. error (with null id)
        let errorNullIdJSON = try ProtocolCodec.encodeResponse(.error(id: nil, message: "parse error"))
        let errorNullIdObj = try parseJSONObject(errorNullIdJSON)
        #expect(Set(errorNullIdObj.keys) == Set(["type", "id", "message"]))
        #expect(errorNullIdObj["type"] as? String == "error")
        #expect(errorNullIdObj["id"] is NSNull)
        #expect(errorNullIdObj["message"] as? String == "parse error")
    }

    // MARK: - End-to-End Line Loop Tests

    @Test("End-to-end line loop over injected FakeTranscriber")
    func endToEndLineLoop() async throws {
        let fake = FakeTranscriber()
        await fake.setFailingWavPaths(["/abs/missing.wav"])
        let session = ProtocolSession(transcriber: fake)

        let inputLines = [
            // 1. Ping works before any model is loaded
            #"{"type":"ping"}"#,
            // 2. Transcribe before load returns error with the request id
            #"{"type":"transcribe","id":1,"wav_path":"/abs/clip.wav"}"#,
            // 3. Load v3 emits progress lines then loaded
            #"{"type":"load","version":"v3","models_dir":"/tmp/models"}"#,
            // 4. Loading already-loaded v3 is a no-op that still returns loaded
            #"{"type":"load","version":"v3","models_dir":"/tmp/models"}"#,
            // 5. Transcribe with tokens (default)
            #"{"type":"transcribe","id":7,"wav_path":"/abs/clip.wav"}"#,
            // 6. Transcribe with tokens: false omits tokens key
            #"{"type":"transcribe","id":8,"wav_path":"/abs/clip.wav","tokens":false}"#,
            // 7. Malformed JSON emits error with id: null and keeps running
            #"{"type":"transcribe", broken"#,
            // 8. Unknown version emits error with id: null and keeps running
            #"{"type":"load","version":"v2","models_dir":"/tmp/models"}"#,
            // 9. Runtime transcription error preserves request id and keeps running
            #"{"type":"transcribe","id":9,"wav_path":"/abs/missing.wav"}"#,
            // 10. Loading ultra replaces v3
            #"{"type":"load","version":"ultra","models_dir":"/tmp/models"}"#,
            // 11. Shutdown emits goodbye and stops loop
            #"{"type":"shutdown"}"#,
            // 12. Trailing line after shutdown is never processed
            #"{"type":"ping"}"#,
        ]

        let outputLines = await session.runLines(inputLines)
        let decodedResponses = try outputLines.map { line -> Response in
            let data = Data(line.utf8)
            return try JSONDecoder().decode(Response.self, from: data)
        }

        #expect(outputLines.count == 15)

        // 1. pong
        #expect(decodedResponses[0] == .pong)

        // 2. error with id 1 (model not loaded)
        if case .error(let id, let message) = decodedResponses[1] {
            #expect(id == 1)
            #expect(message.contains("load"))
        } else {
            Issue.record("Expected error response at index 1")
        }

        // 3. load v3 -> 2 progress lines + 1 loaded line
        #expect(decodedResponses[2] == .progress(stage: .download, fraction: 0.5))
        #expect(decodedResponses[3] == .progress(stage: .compile, fraction: 1.0))
        #expect(decodedResponses[4] == .loaded(version: .v3, loadSecs: 1.25))

        // 4. second load v3 -> immediate loaded (no progress lines, no extra load call)
        #expect(decodedResponses[5] == .loaded(version: .v3, loadSecs: 0.0))

        // 5. transcribe id 7 (with tokens)
        if case .result(let payload) = decodedResponses[6] {
            #expect(payload.id == 7)
            #expect(payload.text == "forecast")
            #expect(payload.words == [WordTimingItem(w: "forecast", start: 12.34, end: 12.80)])
            #expect(payload.tokens?.count == 2)
        } else {
            Issue.record("Expected result response at index 6")
        }

        // 6. transcribe id 8 (tokens: false -> tokens omitted from JSON)
        if case .result(let payload) = decodedResponses[7] {
            #expect(payload.id == 8)
            #expect(payload.tokens == nil)
            let rawObj = try parseJSONObject(outputLines[7])
            #expect(rawObj["tokens"] == nil)
        } else {
            Issue.record("Expected result response at index 7")
        }

        // 7. malformed JSON -> error with id nil
        if case .error(let id, _) = decodedResponses[8] {
            #expect(id == nil)
            let rawObj = try parseJSONObject(outputLines[8])
            #expect(rawObj["id"] is NSNull)
        } else {
            Issue.record("Expected error response at index 8")
        }

        // 8. unknown version -> error with id nil
        if case .error(let id, let msg) = decodedResponses[9] {
            #expect(id == nil)
            #expect(msg.contains("v2"))
        } else {
            Issue.record("Expected error response at index 9")
        }

        // 9. missing file -> error with id 9
        if case .error(let id, let msg) = decodedResponses[10] {
            #expect(id == 9)
            #expect(msg.contains("/abs/missing.wav"))
        } else {
            Issue.record("Expected error response at index 10")
        }

        // 10. load ultra -> 2 progress lines + loaded ultra
        #expect(decodedResponses[11] == .progress(stage: .download, fraction: 0.5))
        #expect(decodedResponses[12] == .progress(stage: .compile, fraction: 1.0))
        #expect(decodedResponses[13] == .loaded(version: .ultra, loadSecs: 1.25))

        // 11. shutdown -> goodbye (and line 12 after shutdown was ignored)
        #expect(decodedResponses[14] == .goodbye)

        // Verify FakeTranscriber was only called twice for load (v3 once, ultra once)
        let recordedLoadCalls = await fake.loadCalls
        #expect(
            recordedLoadCalls == [
                FakeTranscriber.LoadCall(version: .v3, modelsDir: "/tmp/models"),
                FakeTranscriber.LoadCall(version: .ultra, modelsDir: "/tmp/models"),
            ]
        )

        let recordedTranscribeCalls = await fake.transcribeCalls
        #expect(
            recordedTranscribeCalls == [
                FakeTranscriber.TranscribeCall(wavPath: "/abs/clip.wav", includeTokens: true),
                FakeTranscriber.TranscribeCall(wavPath: "/abs/clip.wav", includeTokens: false),
                FakeTranscriber.TranscribeCall(wavPath: "/abs/missing.wav", includeTokens: true),
            ]
        )
    }

    @Test("First load throws error, session keeps running, second load succeeds and transcribe returns result")
    func loadFailureRecoveryInLoop() async throws {
        let fake = FakeTranscriber()
        await fake.setRemainingLoadFailures(1)
        let session = ProtocolSession(transcriber: fake)

        let inputLines = [
            // 1. First load throws -> error response with id: null, session stays alive
            #"{"type":"load","version":"v3","models_dir":"/tmp/models"}"#,
            // 2. Second load of the same version succeeds (was not falsely marked loaded)
            #"{"type":"load","version":"v3","models_dir":"/tmp/models"}"#,
            // 3. Subsequent transcribe succeeds and returns result
            #"{"type":"transcribe","id":42,"wav_path":"/abs/clip.wav"}"#,
            // 4. Shutdown
            #"{"type":"shutdown"}"#,
        ]

        let outputLines = await session.runLines(inputLines)
        let decodedResponses = try outputLines.map { line -> Response in
            try JSONDecoder().decode(Response.self, from: Data(line.utf8))
        }

        #expect(decodedResponses.count == 6)

        // 1. Error from first load attempt
        if case .error(let id, let message) = decodedResponses[0] {
            #expect(id == nil)
            #expect(message.contains("Simulated transient load failure"))
        } else {
            Issue.record("Expected error response from failed first load")
        }

        // 2. Second load succeeds -> progress + loaded
        #expect(decodedResponses[1] == .progress(stage: .download, fraction: 0.5))
        #expect(decodedResponses[2] == .progress(stage: .compile, fraction: 1.0))
        #expect(decodedResponses[3] == .loaded(version: .v3, loadSecs: 1.25))

        // 3. Subsequent transcribe returns result for id 42
        if case .result(let payload) = decodedResponses[4] {
            #expect(payload.id == 42)
            #expect(payload.text == "forecast")
        } else {
            Issue.record("Expected result response after successful second load")
        }

        // 4. Goodbye
        #expect(decodedResponses[5] == .goodbye)

        let recordedLoadCalls = await fake.loadCalls
        #expect(
            recordedLoadCalls == [
                FakeTranscriber.LoadCall(version: .v3, modelsDir: "/tmp/models"),
                FakeTranscriber.LoadCall(version: .v3, modelsDir: "/tmp/models"),
            ]
        )
    }

    @Test("Status works before any load, rejects unknown version with error, and keeps loop running")
    func statusBeforeLoadAndUnknownVersion() async throws {
        let fake = FakeTranscriber()
        await fake.setDownloadedVersions([.ultra])
        let session = ProtocolSession(transcriber: fake)

        let inputLines = [
            // 1. Status before any model is loaded (v3 not downloaded)
            #"{"type":"status","version":"v3","models_dir":"/tmp/models"}"#,
            // 2. Status before any model is loaded (ultra downloaded)
            #"{"type":"status","version":"ultra","models_dir":"/tmp/models"}"#,
            // 3. Status with an unknown version returns error and loop keeps running
            #"{"type":"status","version":"v2","models_dir":"/tmp/models"}"#,
            // 4. Transcribe still fails because status did not load any model
            #"{"type":"transcribe","id":5,"wav_path":"/abs/clip.wav"}"#,
            // 5. Load v3 marks v3 downloaded and loads it
            #"{"type":"load","version":"v3","models_dir":"/tmp/models"}"#,
            // 6. Status for v3 now reports downloaded: true
            #"{"type":"status","version":"v3","models_dir":"/tmp/models"}"#,
            // 7. Shutdown
            #"{"type":"shutdown"}"#,
        ]

        let outputLines = await session.runLines(inputLines)
        let decodedResponses = try outputLines.map { line -> Response in
            try JSONDecoder().decode(Response.self, from: Data(line.utf8))
        }

        #expect(decodedResponses.count == 9)

        // 1. status v3 (downloaded: false)
        #expect(
            decodedResponses[0]
                == .status(version: .v3, downloaded: false, path: "/tmp/models/parakeet-tdt-0.6b-v3")
        )

        // 2. status ultra (downloaded: true)
        #expect(
            decodedResponses[1]
                == .status(version: .ultra, downloaded: true, path: "/tmp/models/parakeet-ultra")
        )

        // 3. unknown version -> error with id: nil
        if case .error(let id, let message) = decodedResponses[2] {
            #expect(id == nil)
            #expect(message.contains("v2"))
        } else {
            Issue.record("Expected error response for unknown status version")
        }

        // 4. transcribe still fails with modelNotLoaded (id 5)
        if case .error(let id, let message) = decodedResponses[3] {
            #expect(id == 5)
            #expect(message.contains("load"))
        } else {
            Issue.record("Expected modelNotLoaded error after status without load")
        }

        // 5. load v3 -> 2 progress lines + loaded
        #expect(decodedResponses[4] == .progress(stage: .download, fraction: 0.5))
        #expect(decodedResponses[5] == .progress(stage: .compile, fraction: 1.0))
        #expect(decodedResponses[6] == .loaded(version: .v3, loadSecs: 1.25))

        // 6. status v3 after load (downloaded: true)
        #expect(
            decodedResponses[7]
                == .status(version: .v3, downloaded: true, path: "/tmp/models/parakeet-tdt-0.6b-v3")
        )

        // 7. goodbye
        #expect(decodedResponses[8] == .goodbye)

        let recordedStatusCalls = await fake.statusCalls
        #expect(
            recordedStatusCalls == [
                FakeTranscriber.StatusCall(version: .v3, modelsDir: "/tmp/models"),
                FakeTranscriber.StatusCall(version: .ultra, modelsDir: "/tmp/models"),
                FakeTranscriber.StatusCall(version: .v3, modelsDir: "/tmp/models"),
            ]
        )
    }

    @Test("Progress normalizer is per-stage monotonic, restarts compile at 0, drops late downloads, and caps below 1.0 until completion")
    func progressNormalizerMonotonicAndCapped() {
        struct Event: Sendable, Equatable {
            let stage: ProgressStage
            let fraction: Double
        }

        let events = OSAllocatedUnfairLock(initialState: [Event]())
        let normalizer = LoadProgressBridge { stage, fraction in
            events.withLock { $0.append(Event(stage: stage, fraction: fraction)) }
        }

        // Initial listing emits 0.0 once.
        normalizer.handleListing()
        normalizer.handleListing()

        // Cache-hit .downloading(0, 0) is ignored.
        normalizer.handleDownloading(fractionCompleted: 0.5, totalFiles: 0)

        // Download progress: sub-1% step is rate-limited; regression is ignored; 1.0 is capped at 0.99.
        normalizer.handleDownloading(fractionCompleted: 0.005, totalFiles: 10)
        normalizer.handleDownloading(fractionCompleted: 0.25, totalFiles: 10)
        normalizer.handleDownloading(fractionCompleted: 0.10, totalFiles: 10)
        normalizer.handleDownloading(fractionCompleted: 0.254, totalFiles: 10)
        normalizer.handleDownloading(fractionCompleted: 0.50, totalFiles: 10)
        normalizer.handleDownloading(fractionCompleted: 1.0, totalFiles: 10)

        // Switching from download (at 0.99) to compile restarts compile at 0.0.
        normalizer.report(stage: .compile, fraction: 0.0)

        // Late download callbacks after compile started must be dropped (no regression to .download, no effect on .compile).
        let countAfterCompileStart = events.withLock { $0.count }
        normalizer.handleListing()
        normalizer.handleDownloading(fractionCompleted: 0.8, totalFiles: 4)
        normalizer.report(stage: .download, fraction: 0.95)
        #expect(events.withLock { $0.count } == countAfterCompileStart)

        // Simulate FluidAudio restarting per-component compile passes across download() and load() (8 models = 16 callbacks,
        // plus extra callbacks beyond expected count to verify the 0.99 cap holds).
        let componentNames = [
            "Preprocessor.mlmodelc",
            "Encoder.mlmodelc",
            "Decoder.mlmodelc",
            "JointDecisionv3.mlmodelc",
        ]
        for pass in 0..<3 {
            for name in componentNames {
                normalizer.handleCompiling(modelName: name)
                normalizer.handleCompiling(modelName: "")
            }
            if pass == 0 {
                // Mid-compile regression and sub-1% step within compile are ignored.
                normalizer.report(stage: .compile, fraction: 0.10)
                normalizer.report(stage: .compile, fraction: 0.504)
            }
        }

        // Late download callbacks after compile reached cap must also be dropped.
        normalizer.handleDownloading(fractionCompleted: 0.1, totalFiles: 4)
        normalizer.report(stage: .download, fraction: 1.0)

        let inProgressEvents = events.withLock { $0 }
        #expect(!inProgressEvents.isEmpty)
        #expect(inProgressEvents.first == Event(stage: .download, fraction: 0.0))

        let downloadEvents = inProgressEvents.filter { $0.stage == .download }
        let compileEvents = inProgressEvents.filter { $0.stage == .compile }
        #expect(!downloadEvents.isEmpty)
        #expect(!compileEvents.isEmpty)
        #expect(downloadEvents.last?.fraction == LoadProgressBridge.inProgressCap)
        #expect(compileEvents.first == Event(stage: .compile, fraction: 0.0))
        #expect(compileEvents.dropFirst().first == Event(stage: .compile, fraction: 1.0 / 16.0))
        #expect(compileEvents.last?.fraction == LoadProgressBridge.inProgressCap)

        // Verify stage ordering (all .download before any .compile) and per-stage monotonic rate-limited fractions < 1.0.
        var previousStage: ProgressStage = .download
        var previousFraction = -1.0
        for event in inProgressEvents {
            #expect(event.fraction >= 0.0)
            #expect(event.fraction <= LoadProgressBridge.inProgressCap)
            #expect(event.fraction < 1.0)
            if event.stage != previousStage {
                #expect(previousStage == .download && event.stage == .compile)
                previousStage = event.stage
                previousFraction = -1.0
            }
            #expect(event.fraction >= previousFraction)
            if previousFraction >= 0 {
                #expect(event.fraction - previousFraction >= LoadProgressBridge.minEmissionDelta - 1e-9)
            }
            previousFraction = event.fraction
        }

        // Final 1.0 is emitted on .compile only when complete() is called right before `loaded`.
        normalizer.complete()
        // Subsequent calls after completion are ignored.
        normalizer.complete()
        normalizer.report(stage: .compile, fraction: 0.5)

        let finalEvents = events.withLock { $0 }
        #expect(finalEvents.count == inProgressEvents.count + 1)
        #expect(finalEvents.last == Event(stage: .compile, fraction: 1.0))
    }

    @Test("One-shot CLI parsing and idle timeout configuration")
    func cliOptionsAndIdleTimeout() throws {
        let loopMode = try CLIOptions.parse(arguments: [], defaultModelsDir: "/default/models")
        #expect(loopMode == .protocolLoop)

        let oneShotDefault = try CLIOptions.parse(
            arguments: ["--transcribe", "/tmp/test.wav"],
            defaultModelsDir: "/default/models"
        )
        #expect(
            oneShotDefault
                == .oneShot(wavPath: "/tmp/test.wav", version: .v3, modelsDir: "/default/models")
        )

        let oneShotCustom = try CLIOptions.parse(
            arguments: ["--transcribe", "/tmp/test.wav", "--version", "ultra", "--models-dir", "/custom/dir"],
            defaultModelsDir: "/default/models"
        )
        #expect(
            oneShotCustom
                == .oneShot(wavPath: "/tmp/test.wav", version: .ultra, modelsDir: "/custom/dir")
        )

        #expect(throws: ProtocolError.unknownVersion("bad")) {
            _ = try CLIOptions.parse(arguments: ["--transcribe", "/tmp/test.wav", "--version", "bad"])
        }

        #expect(CLIOptions.resolveIdleTimeoutSeconds(environment: [:]) == 300.0)
        #expect(CLIOptions.resolveIdleTimeoutSeconds(environment: ["FLUID_HELPER_IDLE_SECS": "45"]) == 45.0)
        #expect(CLIOptions.resolveIdleTimeoutSeconds(environment: ["FLUID_HELPER_IDLE_SECS": "invalid"]) == 300.0)
        #expect(CLIOptions.resolveIdleTimeoutSeconds(environment: ["FLUID_HELPER_IDLE_SECS": "0"]) == 300.0)

        let t0 = ContinuousClock.now
        let tracker = ActivityTracker(now: t0)
        #expect(!tracker.isIdleExpired(timeoutSeconds: 10, now: t0.advanced(by: .seconds(5))))
        #expect(tracker.isIdleExpired(timeoutSeconds: 10, now: t0.advanced(by: .seconds(10))))

        tracker.beginRequest(now: t0.advanced(by: .seconds(9)))
        #expect(!tracker.isIdleExpired(timeoutSeconds: 10, now: t0.advanced(by: .seconds(100))))
        tracker.endRequest(now: t0.advanced(by: .seconds(100)))
        #expect(!tracker.isIdleExpired(timeoutSeconds: 10, now: t0.advanced(by: .seconds(105))))
        #expect(tracker.isIdleExpired(timeoutSeconds: 10, now: t0.advanced(by: .seconds(110))))
    }

    // MARK: - Diarize CLI Parsing & Post-Processing Tests

    @Test("Diarize CLI subcommand parsing and validation")
    func diarizeCLIParsing() throws {
        let diarizeAuto = try CLIOptions.parse(
            arguments: ["diarize", "--audio", "/tmp/meeting.wav", "--models-dir", "/tmp/diar-models"]
        )
        #expect(
            diarizeAuto
                == .diarize(
                    audioPath: "/tmp/meeting.wav",
                    modelsDir: "/tmp/diar-models",
                    numSpeakers: nil
                )
        )

        let diarizeFixedSpeakers = try CLIOptions.parse(
            arguments: [
                "diarize",
                "--audio", "/tmp/meeting.wav",
                "--models-dir", "/tmp/diar-models",
                "--num-speakers", "3",
            ]
        )
        #expect(
            diarizeFixedSpeakers
                == .diarize(
                    audioPath: "/tmp/meeting.wav",
                    modelsDir: "/tmp/diar-models",
                    numSpeakers: 3
                )
        )

        #expect(throws: ProtocolError.missingField("--audio")) {
            _ = try CLIOptions.parse(arguments: ["diarize", "--models-dir", "/tmp/diar-models"])
        }

        #expect(throws: ProtocolError.missingField("--models-dir")) {
            _ = try CLIOptions.parse(arguments: ["diarize", "--audio", "/tmp/meeting.wav"])
        }

        #expect(
            throws: ProtocolError.invalidField(
                field: "--num-speakers",
                reason: "must be a positive integer"
            )
        ) {
            _ = try CLIOptions.parse(
                arguments: [
                    "diarize",
                    "--audio", "/tmp/meeting.wav",
                    "--models-dir", "/tmp/diar-models",
                    "--num-speakers", "0",
                ]
            )
        }

        #expect(
            throws: ProtocolError.invalidField(
                field: "--num-speakers",
                reason: "must be a positive integer"
            )
        ) {
            _ = try CLIOptions.parse(
                arguments: [
                    "diarize",
                    "--audio", "/tmp/meeting.wav",
                    "--models-dir", "/tmp/diar-models",
                    "--num-speakers", "abc",
                ]
            )
        }

        #expect(
            throws: ProtocolError.invalidField(
                field: "--bogus",
                reason: "unknown command-line argument"
            )
        ) {
            _ = try CLIOptions.parse(
                arguments: [
                    "diarize",
                    "--audio", "/tmp/meeting.wav",
                    "--models-dir", "/tmp/diar-models",
                    "--bogus",
                ]
            )
        }
    }

    @Test("Diarization speaker-id mapping, duration-weighted L2-normalised centroids, and JSON output encoding")
    func diarizationPostProcessingAndEncoding() throws {
        // Empty input produces empty output.
        let emptyOutput = DiarizationPostProcessor.buildOutput(from: [])
        #expect(emptyOutput == DiarizeOutput(segments: [], numSpeakers: 0, centroids: []))

        // Input segments out of chronological order:
        // - "S2" appears earliest at [0.5, 1.5] (duration 1.0, embedding [3, 0]) and later at [4.0, 7.0] (duration 3.0, embedding [0, 4])
        //   Weighted mean for "S2": (1.0 * [3, 0] + 3.0 * [0, 4]) / 4.0 = [0.75, 3.0]
        //   L2 norm of [3, 12] is sqrt(9 + 144) = sqrt(153); let's use exact 3-4-5 triangle weights:
        //   Segment 1 for "S2": [0.5, 1.5] (duration 1.0), embedding [3.0, 0.0]
        //   Segment 2 for "S2": [4.0, 6.0] (duration 2.0), embedding [0.0, 2.0]
        //   Duration-weighted sum = 1.0 * [3, 0] + 2.0 * [0, 2] = [3.0, 4.0]
        //   Duration-weighted mean = [1.0, 4.0/3.0], L2-normalised = [0.6, 0.8], clean_secs = 3.0.
        // - "S1" appears second at [2.0, 3.5] (duration 1.5, embedding [0.0, -5.0])
        //   L2-normalised = [0.0, -1.0], clean_secs = 1.5.
        let rawSegments = [
            RawSpeakerSegment(
                speakerId: "S1",
                startTimeSeconds: 2.0,
                endTimeSeconds: 3.5,
                embedding: [0.0, -5.0]
            ),
            RawSpeakerSegment(
                speakerId: "S2",
                startTimeSeconds: 4.0,
                endTimeSeconds: 6.0,
                embedding: [0.0, 2.0]
            ),
            RawSpeakerSegment(
                speakerId: "S2",
                startTimeSeconds: 0.5,
                endTimeSeconds: 1.5,
                embedding: [3.0, 0.0]
            ),
        ]

        let output = DiarizationPostProcessor.buildOutput(from: rawSegments)
        #expect(output.numSpeakers == 2)
        #expect(
            output.segments == [
                DiarizeSegmentItem(start: 0.5, end: 1.5, speaker: 0, voice: false),
                DiarizeSegmentItem(start: 2.0, end: 3.5, speaker: 1, voice: false),
                DiarizeSegmentItem(start: 4.0, end: 6.0, speaker: 0, voice: false),
            ]
        )

        #expect(output.centroids.count == 2)
        #expect(output.centroids[0].speaker == 0)
        #expect(abs(output.centroids[0].cleanSecs - 3.0) < 1e-5)
        #expect(output.centroids[0].embedding.count == 2)
        #expect(abs(output.centroids[0].embedding[0] - 0.6) < 1e-5)
        #expect(abs(output.centroids[0].embedding[1] - 0.8) < 1e-5)

        #expect(output.centroids[1].speaker == 1)
        #expect(abs(output.centroids[1].cleanSecs - 1.5) < 1e-5)
        #expect(output.centroids[1].embedding.count == 2)
        #expect(abs(output.centroids[1].embedding[0] - 0.0) < 1e-5)
        #expect(abs(output.centroids[1].embedding[1] - (-1.0)) < 1e-5)

        // Verify JSON schema matches diarize-helper output format.
        let jsonLine = try ProtocolCodec.encodeDiarizeOutput(output)
        let obj = try parseJSONObject(jsonLine)
        #expect(Set(obj.keys) == Set(["segments", "num_speakers", "centroids"]))
        #expect(obj["num_speakers"] as? Int == 2)

        let segmentsArray = try #require(obj["segments"] as? [[String: Any]])
        #expect(segmentsArray.count == 3)
        #expect(Set(segmentsArray[0].keys) == Set(["start", "end", "speaker", "voice"]))
        #expect(segmentsArray[0]["speaker"] as? Int == 0)
        #expect(segmentsArray[0]["voice"] as? Bool == false)

        let centroidsArray = try #require(obj["centroids"] as? [[String: Any]])
        #expect(centroidsArray.count == 2)
        #expect(Set(centroidsArray[0].keys) == Set(["speaker", "clean_secs", "embedding"]))
        #expect(centroidsArray[0]["speaker"] as? Int == 0)
    }

    // MARK: - Helpers

    private func parseJSONObject(_ jsonLine: String) throws -> [String: Any] {
        #expect(!jsonLine.contains("\n"))
        let data = Data(jsonLine.utf8)
        let raw = try JSONSerialization.jsonObject(with: data)
        return try #require(raw as? [String: Any])
    }
}
