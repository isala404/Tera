use crate::ingest::IngestType;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

/// A fictional versioning CLI
#[derive(Debug, Parser)] // requires `derive` feature
#[command(name = "Tera")]
#[command(about = "Tera is AI assistant which is tailored just for you", long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Ask a question
    #[command(arg_required_else_help = true)]
    Ask {
        /// The question to ask
        query: String,
    },
    /// Let Tera learn from your content
    Upload {
        #[arg(value_name = "Type")]
        content_type: IngestType,
        path: PathBuf,
    },
    /// Tell Tera something to remember
    Remember {
        /// The content to remember
        content: String,
    },
    /// Forget something Tera remembers
    Forget {
        /// The content to forget
        #[arg(group = "forget")]
        content_id: Option<String>,
        /// Forget all content
        #[arg(short, long, group = "forget", default_value = "false")]
        all: bool,
    },
    /// List all content Tera remembers sorted by added date
    List {
        /// How many items you want to skip from the beginning
        #[arg(short, long, default_value = "0")]
        start: u16,
        /// How many items you want to get
        #[arg(short, long, default_value = "10")]
        limit: u16,
    }
}
