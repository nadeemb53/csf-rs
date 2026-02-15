//! Build script to download model files from HuggingFace

use std::env;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::PathBuf;

const MODEL_REPO: &str = "sentence-transformers/all-MiniLM-L6-v2";

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let out_path = PathBuf::from(out_dir);

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    println!("Downloading tokenizer from HuggingFace...");

    let files = vec![
        ("tokenizer.json", "tokenizer.json"),
        ("config.json", "config.json"),
        ("model.safetensors", "model.safetensors"),
    ];

    for (url_name, file_name) in files {
        let url = format!("https://huggingface.co/{}/resolve/main/{}", MODEL_REPO, url_name);
        let path = out_path.join(file_name);
        download_file(&url, &path);
        println!("cargo:rerun-if-changed={}", path.display());
    }

    // Read tokenizer.json
    let tokenizer_path = out_path.join("tokenizer.json");
    let mut tokenizer_file = File::open(&tokenizer_path).expect("Failed to open tokenizer.json");
    let mut tokenizer_bytes = Vec::new();
    tokenizer_file.read_to_end(&mut tokenizer_bytes).expect("Failed to read tokenizer.json");

    // Read config.json
    let config_path = out_path.join("config.json");
    let mut config_file = File::open(&config_path).expect("Failed to open config.json");
    let mut config_bytes = Vec::new();
    config_file.read_to_end(&mut config_bytes).expect("Failed to read config.json");

    // Read model file
    let model_path = out_path.join("model.safetensors");
    let model_data = fs::read(&model_path).expect("Failed to read model.safetensors");
    let model_size = model_data.len();

    // Generate tokenizer_data.rs
    let mut gen_path = manifest_dir;
    gen_path.push("src");
    gen_path.push("tokenizer_data.rs");

    let mut f = File::create(&gen_path).expect("Failed to create tokenizer_data.rs");

    writeln!(f, "/* Auto-generated tokenizer and model data */").unwrap();
    writeln!(f, "/* Downloaded from: {} */", MODEL_REPO).unwrap();
    writeln!(f, "/* Model size: {} bytes */", model_size).unwrap();
    writeln!(f, "").unwrap();

    // Write tokenizer JSON
    writeln!(f, "/// Tokenizer data from HuggingFace (tokenizer.json)").unwrap();
    write!(f, "pub static TOKENIZER_JSON: &[u8] = &[").unwrap();
    for (i, &byte) in tokenizer_bytes.iter().enumerate() {
        if i > 0 {
            write!(f, ", {}", byte).unwrap();
        } else {
            write!(f, "{}", byte).unwrap();
        }
    }
    writeln!(f, "];").unwrap();
    writeln!(f, "").unwrap();

    // Write config JSON
    writeln!(f, "/// Model config from HuggingFace (config.json)").unwrap();
    write!(f, "pub static CONFIG_JSON: &[u8] = &[").unwrap();
    for (i, &byte) in config_bytes.iter().enumerate() {
        if i > 0 {
            write!(f, ", {}", byte).unwrap();
        } else {
            write!(f, "{}", byte).unwrap();
        }
    }
    writeln!(f, "];").unwrap();
    writeln!(f, "").unwrap();

    // Write model data
    writeln!(f, "/// Model weights from HuggingFace (model.safetensors)").unwrap();
    writeln!(f, "pub static MODEL_DATA: &[u8] = &[").unwrap();
    for (i, &byte) in model_data.iter().enumerate() {
        if i > 0 && i % 16 == 0 {
            writeln!(f, "").unwrap();
        }
        write!(f, "{}", byte).unwrap();
        if i < model_data.len() - 1 {
            write!(f, ", ").unwrap();
        }
    }
    writeln!(f, "];").unwrap();

    println!("Generated tokenizer_data.rs ({} bytes tokenizer, {} bytes config, {} bytes model)",
        tokenizer_bytes.len(), config_bytes.len(), model_size);
}

fn download_file(url: &str, path: &PathBuf) {
    if path.exists() {
        let metadata = fs::metadata(path).unwrap();
        if metadata.len() > 100 {
            println!("Using cached: {}", path.display());
            return;
        }
    }

    println!("Downloading: {}", url);

    let output = std::process::Command::new("curl")
        .args(["-L", "-A", "Mozilla/5.0", "-o", path.to_str().unwrap(), url])
        .output()
        .expect("Failed to run curl");

    if !output.status.success() {
        panic!("Failed to download {}: {}", url, String::from_utf8_lossy(&output.stderr));
    }

    let size = fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    if size < 100 {
        panic!("Download failed for {}: got {} bytes", url, size);
    }

    println!("Downloaded: {} ({} bytes)", path.display(), size);
}
