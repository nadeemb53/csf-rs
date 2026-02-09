//! CFS Inference Mobile - Local LLM inference for mobile devices
//!
//! Provides local LLM generation using llama.cpp (via llama-cpp-2 bindings).
//! Designed for efficient on-device inference with GGUF quantized models.

use cfs_core::{CfsError, Result};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::token::data_array::LlamaTokenDataArray;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tracing::info;

/// Local LLM generator using llama.cpp
pub struct LocalGenerator {
    model_path: PathBuf,
    backend: Arc<Mutex<Option<LlamaBackend>>>,
    model: Arc<Mutex<Option<LlamaModel>>>,
}

impl LocalGenerator {
    /// Create a new local generator with the specified GGUF model path
    pub fn new(model_path: PathBuf) -> Self {
        Self {
            model_path,
            backend: Arc::new(Mutex::new(None)),
            model: Arc::new(Mutex::new(None)),
        }
    }

    /// Initialize the model (lazy initialization)
    pub fn initialize(&self) -> Result<()> {
        let mut backend_guard = self
            .backend
            .lock()
            .map_err(|_| CfsError::Inference("Backend mutex poisoned".into()))?;

        if backend_guard.is_some() {
            return Ok(()); // Already initialized
        }

        println!(
            "[LocalGenerator] Initializing llama.cpp with model: {:?}",
            self.model_path
        );
        info!(
            "[LocalGenerator] Initializing llama.cpp with model: {:?}",
            self.model_path
        );

        // Check if file exists
        if !self.model_path.exists() {
            let msg = format!("Model file not found: {:?}", self.model_path);
            println!("[LocalGenerator] {}", msg);
            return Err(CfsError::Inference(msg));
        }

        println!("[LocalGenerator] Model file exists, size: {} bytes",
            std::fs::metadata(&self.model_path).map(|m| m.len()).unwrap_or(0));

        // Initialize backend
        println!("[LocalGenerator] Initializing backend...");
        let backend = LlamaBackend::init()
            .map_err(|e| {
                let msg = format!("Failed to init llama backend: {}", e);
                println!("[LocalGenerator] {}", msg);
                CfsError::Inference(msg)
            })?;
        println!("[LocalGenerator] Backend initialized");

        // Load model with mobile-optimized params
        println!("[LocalGenerator] Loading model...");
        let model_params = LlamaModelParams::default();

        let model = LlamaModel::load_from_file(&backend, &self.model_path, &model_params)
            .map_err(|e| {
                let msg = format!("Failed to load model: {}", e);
                println!("[LocalGenerator] {}", msg);
                CfsError::Inference(msg)
            })?;
        println!("[LocalGenerator] Model loaded");

        // Store initialized components
        *backend_guard = Some(backend);

        let mut model_guard = self
            .model
            .lock()
            .map_err(|_| CfsError::Inference("Model mutex poisoned".into()))?;
        *model_guard = Some(model);

        println!("[LocalGenerator] Initialization complete!");
        info!("[LocalGenerator] Model loaded successfully");
        Ok(())
    }

    /// Generate text from a prompt
    pub fn generate(&self, prompt: &str, max_tokens: u32) -> Result<String> {
        // Ensure model is initialized
        self.initialize()?;

        let backend_guard = self
            .backend
            .lock()
            .map_err(|_| CfsError::Inference("Backend mutex poisoned".into()))?;
        let backend = backend_guard
            .as_ref()
            .ok_or_else(|| CfsError::Inference("Backend not initialized".into()))?;

        let model_guard = self
            .model
            .lock()
            .map_err(|_| CfsError::Inference("Model mutex poisoned".into()))?;
        let model = model_guard
            .as_ref()
            .ok_or_else(|| CfsError::Inference("Model not initialized".into()))?;

        // Create context with mobile-friendly settings
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(2048)) // Smaller context for mobile
            .with_n_batch(512);

        let mut ctx = model
            .new_context(backend, ctx_params)
            .map_err(|e| CfsError::Inference(format!("Failed to create context: {}", e)))?;

        // Tokenize input
        let tokens = model
            .str_to_token(prompt, llama_cpp_2::model::AddBos::Always)
            .map_err(|e| CfsError::Inference(format!("Tokenization failed: {}", e)))?;

        info!(
            "[LocalGenerator] Input tokens: {}, generating up to {} tokens",
            tokens.len(),
            max_tokens
        );

        // Create batch and add tokens
        let mut batch = LlamaBatch::new(2048, 1);
        for (i, token) in tokens.iter().enumerate() {
            let is_last = i == tokens.len() - 1;
            batch
                .add(*token, i as i32, &[0], is_last)
                .map_err(|e| CfsError::Inference(format!("Failed to add token to batch: {}", e)))?;
        }

        // Decode initial batch
        ctx.decode(&mut batch)
            .map_err(|e| CfsError::Inference(format!("Initial decode failed: {}", e)))?;

        // Generate tokens
        let mut output_tokens = Vec::new();
        let mut n_cur = tokens.len();

        for _ in 0..max_tokens {
            // Get logits for the last token
            let candidates = ctx.candidates_ith(batch.n_tokens() - 1);
            let mut candidates_data = LlamaTokenDataArray::from_iter(candidates, false);

            // Sample greedily (use the token data array's method)
            let new_token = candidates_data.sample_token_greedy();

            // Check for EOS
            if model.is_eog_token(new_token) {
                break;
            }

            output_tokens.push(new_token);

            // Prepare next batch
            batch.clear();
            batch
                .add(new_token, n_cur as i32, &[0], true)
                .map_err(|e| CfsError::Inference(format!("Failed to add generated token: {}", e)))?;

            n_cur += 1;

            // Decode
            ctx.decode(&mut batch)
                .map_err(|e| CfsError::Inference(format!("Decode step failed: {}", e)))?;
        }

        // Convert tokens back to string using token_to_piece_bytes
        let mut output = String::new();
        for token in &output_tokens {
            match model.token_to_piece_bytes(*token, 32, true, None) {
                Ok(bytes) => {
                    output.push_str(&String::from_utf8_lossy(&bytes));
                }
                Err(e) => {
                    return Err(CfsError::Inference(format!(
                        "Token to string failed: {:?}",
                        e
                    )));
                }
            }
        }

        info!("[LocalGenerator] Generated {} tokens", output_tokens.len());

        Ok(output)
    }
}

/// Implement cfs_query::LLMGenerator for LocalGenerator
#[async_trait::async_trait]
impl cfs_query::LLMGenerator for LocalGenerator {
    async fn generate(&self, prompt: &str) -> Result<String> {
        // Clone prompt to owned String for 'static lifetime requirement
        let prompt_owned = prompt.to_string();

        // Run sync generation in blocking task to not block async runtime
        let generator = LocalGenerator {
            model_path: self.model_path.clone(),
            backend: self.backend.clone(),
            model: self.model.clone(),
        };

        tokio::task::spawn_blocking(move || generator.generate(&prompt_owned, 256))
            .await
            .map_err(|e| CfsError::Inference(format!("Generation task failed: {}", e)))?
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generator_creation() {
        let gen = LocalGenerator::new("/path/to/model.gguf".into());
        assert_eq!(gen.model_path, PathBuf::from("/path/to/model.gguf"));
    }
}
