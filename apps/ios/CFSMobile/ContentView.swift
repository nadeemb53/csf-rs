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
    
    @State private var showingSettings = false
    
    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // 1. Search Bar at Top
                VStack(spacing: 8) {
                    TextField("Search files or consult substrate...", text: $queryText)
                        .textFieldStyle(.roundedBorder)
                        .padding(.horizontal)
                        .autocorrectionDisabled()
                        .textInputAutocapitalization(.none)
                        .onSubmit { runQuery() }

                    HStack(spacing: 12) {
                        Button(action: runQuery) {
                            Text("Search")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.bordered)
                        .disabled(isSearching || queryText.isEmpty)

                        Button(action: askAi) {
                            HStack {
                                if isSearching && aiAnswer.isEmpty {
                                    ProgressView().tint(.white)
                                } else {
                                    Image(systemName: "sparkles")
                                    Text("Consult Substrate")
                                }
                            }
                            .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.borderedProminent)
                        .tint(.indigo)
                        .disabled(isSearching || queryText.isEmpty)
                    }
                    .padding(.horizontal)
                }
                .padding(.vertical, 12)
                .background(Color(.systemBackground))
                .shadow(color: Color.black.opacity(0.05), radius: 5, y: 5)

                ScrollView {
                    VStack(alignment: .leading, spacing: 20) {
                        // 2. Privacy & Determinism Badge (New)
                        HStack {
                            Label("End-to-End Private", systemImage: "lock.shield.fill")
                                .font(.caption.bold())
                                .foregroundColor(.green)
                            Spacer()
                            NavigationLink(destination: VerificationView(stateRoot: stateRoot)) {
                                HStack(spacing: 4) {
                                    Text("Verify State")
                                    Image(systemName: "checkmark.seal")
                                }
                                .font(.caption.bold())
                                .foregroundColor(.indigo)
                            }
                        }
                        .padding(.horizontal)
                        .padding(.top, 8)

                        // 3. AI Response Section
                        if !aiAnswer.isEmpty || (isSearching && aiAnswer.isEmpty) {
                            VStack(alignment: .leading, spacing: 12) {
                                HStack {
                                    Label("Synthesized Insight", systemImage: "sparkles")
                                        .font(.headline)
                                        .foregroundColor(.indigo)
                                    Spacer()
                                    if !aiMetrics.isEmpty {
                                        Text(aiMetrics)
                                            .font(.system(.caption2, design: .monospaced))
                                            .foregroundColor(.secondary)
                                    }
                                }

                                if isSearching && aiAnswer.isEmpty {
                                    HStack {
                                        ProgressView()
                                        Text("Extracting verified chunks...")
                                            .font(.subheadline)
                                            .foregroundColor(.secondary)
                                    }
                                    .padding(.vertical)
                                } else {
                                    Text(aiAnswer)
                                        .font(.body)
                                        .lineSpacing(4)
                                        .textSelection(.enabled)
                                        .fixedSize(horizontal: false, vertical: false)
                                }
                            }
                            .padding()
                            .background(Color.indigo.opacity(0.05))
                            .cornerRadius(12)
                            .overlay(
                                RoundedRectangle(cornerRadius: 12)
                                    .stroke(Color.indigo.opacity(0.1), lineWidth: 1)
                            )
                            .padding(.horizontal)
                        }

                        // Status message
                        if !syncStatus.isEmpty && syncStatus != "Ready" {
                            Text(syncStatus)
                                .font(.caption)
                                .foregroundColor(.secondary)
                                .padding(.horizontal)
                        }

                        // 4. Search Results
                        if !results.isEmpty {
                            VStack(alignment: .leading, spacing: 12) {
                                Text("Relevant Files")
                                    .font(.headline)
                                    .padding(.horizontal)

                                ForEach(results) { res in
                                    VStack(alignment: .leading, spacing: 6) {
                                        Text(res.text)
                                            .font(.subheadline)
                                            .lineLimit(4)
                                        
                                        HStack {
                                            Image(systemName: "doc.text")
                                            Text(res.doc_path.components(separatedBy: "/").last ?? res.doc_path)
                                            Spacer()
                                            Text("Score: \(String(format: "%.2f", res.score))")
                                        }
                                        .font(.system(size: 10))
                                        .foregroundColor(.blue)
                                    }
                                    .padding()
                                    .background(Color.secondary.opacity(0.05))
                                    .cornerRadius(8)
                                    .padding(.horizontal)
                                }
                            }
                        }
                    }
                    .padding(.bottom, 20)
                }
            }
            .navigationTitle("CFS Mobile")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: { showingSettings = true }) {
                        Image(systemName: "gear")
                    }
                }
            }
            .sheet(isPresented: $showingSettings) {
                SettingsView(relayUrl: $relayUrl, stateRoot: stateRoot, stats: stats, onSync: pullSync)
            }
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
                        // First test if llama.cpp backend works at all
                        print("[CFS-iOS] Testing llama.cpp backend...")
                        let backendTest = self.bridge.testLlmBackend()
                        print("[CFS-iOS] Backend test result: \(backendTest)")

                        if backendTest != 0 {
                            let error = self.bridge.getLastError()
                            continuation.resume(returning: (nil, 0, "Backend test failed: \(error)"))
                            return
                        }
                        print("[CFS-iOS] Backend test passed!")

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

                        print("[CFS-iOS] Initializing LLM with model...")
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
                self.syncStatus = "Substrate synthesis complete"
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
} // End of ContentView

struct VerificationView: View {
    let stateRoot: String

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Trust & Transparency")
                        .font(.largeTitle.bold())
                    Text("CFS ensures your data is private, deterministic, and verifiable.")
                        .foregroundColor(.secondary)
                }

                VStack(alignment: .leading, spacing: 16) {
                    Label("Deterministic State", systemImage: "circle.grid.hex.fill")
                        .font(.headline)
                    Text("The State Root represents the mathematical 'thumbprint' of your entire knowledge graph. Because CFS is deterministic, identical files will always produce the exact same root across any device.")
                        .font(.subheadline)
                    
                    Text(stateRoot)
                        .font(.system(.caption, design: .monospaced))
                        .padding()
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color.secondary.opacity(0.1))
                        .cornerRadius(8)
                }

                Divider()

                VStack(alignment: .leading, spacing: 16) {
                    Label("Detachable Intelligence", systemImage: "cpu")
                        .font(.headline)
                    Text("Intelligence runs entirely on this device. It is a read-only lens over your verified substrate. Your queries and file contents never leave your hardware.")
                        .font(.subheadline)
                }

                Divider()

                VStack(alignment: .leading, spacing: 16) {
                    Label("Verifiable Sync", systemImage: "arrow.left.and.right.circle.fill")
                        .font(.headline)
                    Text("When you sync with your Mac, CFS only exchanges encrypted 'diffs'. The State Root is used to verify that both devices have perfectly synchronized states without needing to trust the relay server.")
                        .font(.subheadline)
                }

                Spacer()
            }
            .padding()
        }
        .navigationTitle("Verification")
        .navigationBarTitleDisplayMode(.inline)
    }
}

struct SettingsView: View {
    @Binding var relayUrl: String
    let stateRoot: String
    let stats: String
    let onSync: () -> Void
    @Environment(\.dismiss) var dismiss

    var body: some View {
        NavigationView {
            List {
                Section("Relay Configuration") {
                    TextField("Relay URL", text: $relayUrl)
                        .autocorrectionDisabled()
                        .textInputAutocapitalization(.none)
                    Button("Trigger Sync", action: onSync)
                }

                Section("System Status") {
                    HStack {
                        Text("State Root")
                        Spacer()
                        Text(stateRoot)
                            .font(.system(.caption, design: .monospaced))
                            .lineLimit(1)
                    }
                    HStack {
                        Text("Database")
                        Spacer()
                        Text(stats)
                            .font(.caption)
                    }
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}

// Wrapper to prevent the thought from being sent to user
struct Vroot<Content: View>: View {
    let content: Content
    init(@ViewBuilder content: () -> Content) { self.content = content() }
    var body: some View { content }
}
