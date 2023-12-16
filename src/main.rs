use std::io::Write;

use crate::{
    cli::{Cli, Commands},
    database::get_releted_chunks,
};
use anyhow::Result;
use clap::Parser;
use embeddings::get_embeddings;
use ingest::{
    ingest_via_audio_file, ingest_via_cli, ingest_via_pdf_file, ingest_via_txt_file,
    ingest_wa_chat_log,
};
use prettytable::{Table, row};
mod cli;
mod database;
mod embeddings;
mod inference;
mod ingest;
mod whisper;

#[tokio::main]
async fn main() -> Result<()> {
    let args = Cli::parse();

    match args.command {
        Commands::Ask { query } => {
            let embeddings: Vec<f32> = get_embeddings(&query)?.reshape((384,))?.to_vec1()?;
            let k = get_releted_chunks(embeddings).await?;
            let mut context = vec![];
            for reference in k.iter() {
                let releted = reference.get_adjacent_chunks(1, 1).await?;
                context.extend(releted);
            }
            let answer = inference::answer_with_context(&query, context).await?;
            println!("Answer: {}", answer);
        }
        Commands::Upload { content_type, path } => match content_type {
            ingest::IngestType::Whatsapp => {
                ingest_wa_chat_log(path).await?;
            }
            ingest::IngestType::Text => {
                ingest_via_txt_file(path).await?;
            }
            ingest::IngestType::PDF => {
                ingest_via_pdf_file(path).await?;
            }
            ingest::IngestType::Audio => {
                ingest_via_audio_file(path).await?;
            }
        },
        Commands::Remember { content } => {
            ingest_via_cli(&content).await?;
        },
        Commands::Forget { content_id, all } => {
            if all {
                print!("Are you sure you want me to forget everything? this cannot be undone! [y/N]: ");
                std::io::stdout().flush()?;
                let mut input = String::new();
                std::io::stdin().read_line(&mut input)?;
                if input.trim().to_lowercase() == "y" {
                    database::forget_all_content().await?;
                    println!("Wiped all content");
                } else {
                    println!("Aborting");
                }
            } else {
                if content_id.is_none() {
                    println!("You need to specify a content id to forget");
                    return Ok(());
                }
                database::delete_content(content_id.clone().unwrap().as_str()).await?;
                println!("Content {} was deleted", content_id.unwrap());
            }
        },
        Commands::List { start, limit } => {
            let content = database::get_all_content(start, limit).await?;
            let mut table = Table::new();
            table.add_row(row!["ID", "Title", "Created At"]);
            for c in content {
                table.add_row(row![c.id.id, c.title, c.created_at]);
            }
            table.printstd();
        }
    }

    Ok(())
}
