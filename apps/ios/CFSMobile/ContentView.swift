import SwiftUI

struct ContentView: View {
    @State private var relayUrl = "http://127.0.0.1:8080"
    @State private var queryText = ""
    @State private var stateRoot = "None"
    @State private var results: [SearchResult] = []
    @State private var syncStatus = "Ready"
    @State private var stats = "Docs: 0, Chunks: 0"
    @State private var isSearching = false
    @State private var aiAnswer = ""
    @State private var aiMetrics = ""
    @State private var isLlmInitialized = false

    // In V0, we use a single instance of the bridge
    private let bridge: CfsBridge

    init() {
        let paths = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
        let dbPath = paths[0].appendingPathComponent("mobile_graph.db").path
        self.bridge = CfsBridge(dbPath: dbPath)
        // LLM initialization is deferred to first use (on background thread)
    }
    
    var body: some View {
        Vroot {
            VStack(alignment: .leading, spacing: 20) {
                Text("CFS iOS (V0)")
                    .font(.largeTitle)
                    .bold()
                
                Group {
                    Text("State Root")
                        .font(.headline)
                    Text(stateRoot)
                        .font(.caption)
                        .monospaced()
                        .padding(8)
                        .background(Color.secondary.opacity(0.1))
                        .cornerRadius(4)

                    HStack {
                        Text(stats)
                        Spacer()
                        if !aiMetrics.isEmpty {
                            Text(aiMetrics)
                                .font(.system(size: 10, weight: .bold, design: .monospaced))
                                .foregroundColor(.orange)
                        }
                    }
                    .font(.caption2)
                    .foregroundColor(.blue)
                }

                if !aiAnswer.isEmpty {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("AI Response")
                            .font(.headline)
                        Text(aiAnswer)
                            .padding()
                            .background(Color.orange.opacity(0.1))
                            .cornerRadius(8)
                    }
                    .transition(.move(edge: .top).combined(with: .opacity))
                }
                
                HStack {
                    VStack(alignment: .leading) {
                        Text("Relay URL")
                            .font(.caption)
                        TextField("http://...", text: $relayUrl)
                            .textFieldStyle(.roundedBorder)
                            .autocorrectionDisabled()
                            .textInputAutocapitalization(.none)
                    }
                    
                    Button(action: {
                        pullSync()
                    }) {
                        Label("Pull", systemImage: "arrow.down.circle")
                    }
                    .buttonStyle(.borderedProminent)
                }
                
                Text(syncStatus)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .lineLimit(2)
                
                Divider()
                
                VStack(alignment: .leading) {
                    TextField("Search chunks...", text: $queryText)
                        .textFieldStyle(.roundedBorder)
                        .autocorrectionDisabled()
                        .textInputAutocapitalization(.none)
                    
                    HStack {
                        Button("Query") {
                            runQuery()
                        }
                        .buttonStyle(.bordered)
                        .disabled(isSearching || queryText.isEmpty)

                        Button("Ask AI") {
                            askAi()
                        }
                        .buttonStyle(.borderedProminent)
                        .tint(.orange)
                        .disabled(isSearching || queryText.isEmpty)

                        if isSearching {
                            ProgressView()
                                .padding(.leading, 8)
                        }
                    }
                }
                
                List(results) { res in
                    VStack(alignment: .leading) {
                        Text("Score: \(String(format: "%.4f", res.score))")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Text(res.text)
                            .font(.body)
                        Text(res.doc_path)
                            .font(.caption2)
                            .foregroundColor(.blue)
                    }
                }
                .listStyle(.plain)
            }
            .padding()
        }
    }
    
    func pullSync() {
        syncStatus = "Syncing..."
        let keyHex = "0101010101010101010101010101010101010101010101010101010101010101"
        
        DispatchQueue.global(qos: .userInitiated).async {
            let diffs = bridge.sync(relayUrl: relayUrl, keyHex: keyHex)
            DispatchQueue.main.async {
                if diffs >= 0 {
                    syncStatus = "Applied \(diffs) diffs"
                    stateRoot = bridge.getStateRoot()
                    updateStats()
                } else {
                    let detail = bridge.getLastError()
                    syncStatus = "Error: \(detail)"
                    if detail.contains("connection refused") || detail.contains("localhost") {
                        syncStatus += " (Tip: Use Mac's IP if on real device)"
                    }
                }
            }
        }
    }
    
    func runQuery() {
        guard !queryText.isEmpty else { return }
        syncStatus = "Searching..."
        isSearching = true

        // Run on background thread
        Task {
            let searchResults = bridge.query(text: queryText)
            let error = bridge.getLastError()

            await MainActor.run {
                self.results = searchResults
                self.isSearching = false
                if searchResults.isEmpty && !error.isEmpty && !error.contains("Mutex") {
                    syncStatus = "Search Error: \(error)"
                } else if searchResults.isEmpty {
                    syncStatus = "No results found"
                } else {
                    syncStatus = "Found \(searchResults.count) results"
                }
            }
        }
    }

    func askAi() {
        guard !queryText.isEmpty else { return }

        aiAnswer = ""
        aiMetrics = ""
        syncStatus = "Thinking..."
        isSearching = true

        // Capture values needed for background work
        let query = queryText
        let needsInit = !isLlmInitialized

        Task {
            // Do LLM work on background queue
            let result: (answer: String?, latency: Int, error: String?) = await withCheckedContinuation { continuation in
                DispatchQueue.global(qos: .userInitiated).async {
                    // Initialize LLM if needed
                    if needsInit {
                        let paths = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
                        var modelPath = paths[0].appendingPathComponent("smollm2.gguf").path

                        if !FileManager.default.fileExists(atPath: modelPath) {
                            if let bundlePath = Bundle.main.path(forResource: "smollm2", ofType: "gguf") {
                                modelPath = bundlePath
                                print("[CFS-iOS] Using bundled model at: \(modelPath)")
                            }
                        } else {
                            print("[CFS-iOS] Using Documents model at: \(modelPath)")
                        }

                        guard FileManager.default.fileExists(atPath: modelPath) else {
                            print("[CFS-iOS] No model file found!")
                            continuation.resume(returning: (nil, 0, "No model file. Add smollm2.gguf to Documents."))
                            return
                        }

                        print("[CFS-iOS] Initializing LLM...")
                        let res = self.bridge.initLlm(modelPath: modelPath)
                        print("[CFS-iOS] LLM init result: \(res)")

                        if res != 0 {
                            let error = self.bridge.getLastError()
                            continuation.resume(returning: (nil, 0, "LLM init failed: \(error)"))
                            return
                        }
                    }

                    // Generate answer
                    print("[CFS-iOS] Generating answer for: \(query)")
                    if let res = self.bridge.generate(query: query) {
                        print("[CFS-iOS] Generation successful: \(res.latency_ms)ms")
                        continuation.resume(returning: (res.answer, res.latency_ms, nil))
                    } else {
                        let error = self.bridge.getLastError()
                        print("[CFS-iOS] Generation failed: \(error)")
                        continuation.resume(returning: (nil, 0, error))
                    }
                }
            }

            // Update UI on main actor
            if let answer = result.answer {
                self.isLlmInitialized = true
                self.aiAnswer = answer
                self.aiMetrics = "\(result.latency)ms"
                self.syncStatus = "AI generated answer"
            } else {
                self.syncStatus = "AI Error: \(result.error ?? "Unknown")"
            }
            self.isSearching = false
        }
    }

    func updateStats() {
        let json = bridge.getStats()
        // Simple parse for V0
        if let data = json.data(using: .utf8),
           let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Int] {
            let docs = dict["documents"] ?? 0
            let chunks = dict["chunks"] ?? 0
            let embs = dict["embeddings"] ?? 0
            stats = "Documents: \(docs), Chunks: \(chunks), Embeddings: \(embs)"
        }
    }
}

// Wrapper to prevent the thought from being sent to user
struct Vroot<Content: View>: View {
    let content: Content
    init(@ViewBuilder content: () -> Content) { self.content = content() }
    var body: some View { content }
}
