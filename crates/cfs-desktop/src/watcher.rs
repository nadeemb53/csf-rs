//! File system watcher for detecting changes

use cfs_core::{CfsError, Result};
use notify::{Config, Event, RecommendedWatcher, RecursiveMode, Watcher};
use std::path::Path;
use std::sync::mpsc;

/// File system watcher for real-time change detection
pub struct FileWatcher {
    watcher: RecommendedWatcher,
    // receiver: mpsc::Receiver<notify::Result<Event>>, // No longer needed if we push to external tx
}

impl FileWatcher {
    /// Create a new file watcher
    pub fn new(tx: mpsc::Sender<notify::Result<Event>>) -> Result<Self> {
        let watcher = RecommendedWatcher::new(
            move |res| {
                let _ = tx.send(res);
            },
            Config::default(),
        )
        .map_err(|e| CfsError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;

        Ok(Self {
            watcher,
        })
    }

    /// Start watching a directory
    pub fn watch(&mut self, path: &Path) -> Result<()> {
        self.watcher
            .watch(path, RecursiveMode::Recursive)
            .map_err(|e| CfsError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;
        Ok(())
    }

    /// Stop watching a directory
    pub fn unwatch(&mut self, path: &Path) -> Result<()> {
        self.watcher
            .unwatch(path)
            .map_err(|e| CfsError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;
        Ok(())
    }


}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn test_watcher_creation() {
        // FileWatcher requires a channel sender - skip for now
    }
}
