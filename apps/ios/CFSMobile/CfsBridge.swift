import Foundation

/// Swift wrapper for the CFS Mobile C FFI
class CfsBridge {
    private var context: OpaquePointer?
    
    init(dbPath: String) {
        self.context = cfs_init(dbPath)
    }
    
    deinit {
        if let context = context {
            cfs_free(context)
        }
    }
    
    func sync(relayUrl: String, keyHex: String) -> Int32 {
        guard let context = context else { return -1 }
        return cfs_sync(context, relayUrl, keyHex)
    }
    
    func query(text: String) -> [SearchResult] {
        guard let context = context else { return [] }
        guard let cJson = cfs_query(context, text) else { return [] }
        defer { cfs_free_string(cJson) }
        
        let json = String(cString: cJson)
        let data = json.data(using: .utf8)!
        return (try? JSONDecoder().decode([SearchResult].self, from: data)) ?? []
    }
    
    func getStateRoot() -> String {
        guard let context = context else { return "None" }
        guard let cStr = cfs_get_state_root(context) else { return "Error" }
        defer { cfs_free_string(cStr) }
        return String(cString: cStr)
    }
    
    func getLastError() -> String {
        guard let cStr = cfs_last_error() else { return "Unknown error" }
        defer { cfs_free_string(cStr) }
        return String(cString: cStr)
    }
    func getStats() -> String {
        guard let context = context else { return "{}" }
        guard let cStr = cfs_stats(context) else { return "{}" }
        defer { cfs_free_string(cStr) }
        return String(cString: cStr)
    }

    /// Initialize the LLM with a GGUF model file
    func initLlm(modelPath: String) -> Int32 {
        guard let context = context else { return -1 }
        return cfs_init_llm(context, modelPath)
    }

    /// Generate an AI answer using RAG
    func generate(query: String) -> GenerationResult? {
        guard let context = context else { return nil }
        guard let cJson = cfs_generate(context, query) else { return nil }
        defer { cfs_free_string(cJson) }

        let json = String(cString: cJson)
        let data = json.data(using: .utf8)!
        return try? JSONDecoder().decode(GenerationResult.self, from: data)
    }
}

struct SearchResult: Codable, Identifiable {
    var id: UUID { UUID() }
    let text: String
    let score: Float
    let doc_path: String
}

struct GenerationResult: Codable {
    let answer: String
    let context: String
    let latency_ms: Int
}

// C FFI Prototypes (Must match cfs-mobile/src/lib.rs)
// Note: In a real project, these are generated in a bridging header.
@_silgen_name("cfs_init")
func cfs_init(_ db_path: UnsafePointer<Int8>) -> OpaquePointer?

@_silgen_name("cfs_init_llm")
func cfs_init_llm(_ ctx: OpaquePointer, _ model_path: UnsafePointer<Int8>) -> Int32

@_silgen_name("cfs_generate")
func cfs_generate(_ ctx: OpaquePointer, _ query: UnsafePointer<Int8>) -> UnsafeMutablePointer<Int8>?

@_silgen_name("cfs_sync")
func cfs_sync(_ ctx: OpaquePointer, _ relay_url: UnsafePointer<Int8>, _ key_hex: UnsafePointer<Int8>) -> Int32

@_silgen_name("cfs_query")
func cfs_query(_ ctx: OpaquePointer, _ query: UnsafePointer<Int8>) -> UnsafeMutablePointer<Int8>?

@_silgen_name("cfs_get_state_root")
func cfs_get_state_root(_ ctx: OpaquePointer) -> UnsafeMutablePointer<Int8>?

@_silgen_name("cfs_stats")
func cfs_stats(_ ctx: OpaquePointer) -> UnsafeMutablePointer<Int8>?

@_silgen_name("cfs_last_error")
func cfs_last_error() -> UnsafeMutablePointer<Int8>?

@_silgen_name("cfs_free_string")
func cfs_free_string(_ s: UnsafeMutablePointer<Int8>)

@_silgen_name("cfs_free")
func cfs_free(_ ctx: OpaquePointer)
