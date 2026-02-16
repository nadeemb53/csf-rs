# Canon Protocol

Canon Protocol is infrastructure for AI memory that you can trust. It's a way for AI agents to store knowledge, verify each other's state, and work together without needing to trust a central authority.

Think of it like git, but for AI knowledge instead of code. In git, you can see exactly what changed, verify the history, and collaborate with others even if you're offline. Canon Protocol brings those same properties to AI knowledge management.

The core idea is simple: **make everything deterministic and verifiable by default.**

Instead of storing data and hoping it's consistent, everything in Canon Protocol is:
- **Content-addressed**: Each piece of knowledge is identified by a cryptographic hash of its content, not by a filename or ID. If the content changes, the address changes.
- **Cryptographically committed**: There's always a "state root"—a single hash that represents everything in the system. If you have the root, you can verify nothing was tampered with.
- **Reproducible**: The same input always produces the same output.
- **Local-first**: Everything runs on your device. Your AI's memory is your data. You can sync it elsewhere.

## What It Looks Like In Practice

Here's what you can do with Canon Protocol:

**Build AI memory that doesn't forget.** Feed documents in, get embeddings out, and everything is automatically chunked, indexed, and stored. When your AI needs context, it searches semantically—not by keywords, but by meaning.

**Verify agent reasoning.** An agent can show its "state root" and another agent can verify it. "You said you read the contract at timestamp X. Prove it. Here's the hash. Here's the Merkle path to the root. Check it yourself."

**Sync across devices with zero trust.** You have an iPad and a Mac. Your AI memory syncs between them. The relay server that moves the data can't read a single byte—it's encrypted with keys that never leave your devices.

**Collaborate with other agents.** Two agents can exchange knowledge, verify each other's contributions, and merge their understanding—all without either having to trust the other.

## What Canon Protocol Isn't

It's not an AI model. It doesn't generate text, write code, or have conversations. It's the substrate that other AI systems build on top of.

It's not a general database. You can't just store arbitrary data and query it however you want. The schema is opinionated toward semantic knowledge management—documents, chunks, embeddings, relationships.

It's not a cloud service by default. Everything runs locally. If you want to sync across devices, you need your own relay or you run one yourself.

It's not finished. This is a working implementation, not a polished product. The specs are there, the core works, but there's more to build.

## Where It Could Go

This is where it gets interesting.

**Verifiable AI Markets.** Imagine agents that sell their knowledge. You want to buy "everything this agent knows about tax law." You can verify exactly what you're getting before paying—the state root proves it.

**Agent Accountability.** AI makes a decision that causes a problem. With Canon Protocol, you can trace back exactly what the agent knew at decision time. Not logs it wrote down—cryptographically proven state.

**Personal AI That You Own.** Your AI assistant understands your files, your emails, your life. With Canon Protocol, that memory is truly yours. You can back it up, verify it, move it to a new device, and no company has access to it.

**Federated AI Networks.** Multiple organizations, each with their own AI, sharing knowledge selectively. Each can verify what others contributed. No single point of trust.

None of this is guaranteed—it's just what's possible when AI memory is built on verifiable foundations.

## The Implementation

Canon Protocol is written in Rust. The implementation is modular:

- **cp-core**: The data models, hashing, and cryptographic primitives
- **cp-graph**: SQLite + HNSW hybrid storage
- **cp-embeddings**: Local embedding generation, CPU-only
- **cp-parser**: Document parsing
- **cp-query**: Hybrid search (semantic + keyword) and context assembly
- **cp-sync**: Merkle diffing and encrypted state synchronization
- **cp-relay-client**: Client for syncing through blind relay servers
- **cp-inference-mobile**: Local LLM inference for on-device AI

The `spec/` directory contains the protocol specifications—exactly how everything works, what the guarantees are, and what conformant implementations must do.

## Getting Started

If you want to play with it:

```bash
# Build everything
cargo build

# Run the tests
cargo test

# Try the CLI (if you want to poke around)
cargo run -p cp-desktop-cli -- --help
```
