use crate::database::{insert_content, insert_vector_index, smart_insert_content};
use crate::whisper::whisper_decode;
use anyhow::Context;
use chrono::{NaiveDateTime, Utc};
use clap::ValueEnum;
use regex::Regex;
use serde_json::json;
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

#[derive(ValueEnum, Copy, Clone, Debug, PartialEq, Eq)]
pub enum IngestType {
    Whatsapp,
    PDF,
    Text,
    Audio,
}

#[derive(Debug)]
pub struct Message {
    pub date: NaiveDateTime,
    pub sender: String,
    pub content: String,
}

pub async fn ingest_wa_chat_log(path: PathBuf) -> anyhow::Result<()> {
    let display = path.display();
    let file = match File::open(&path) {
        Err(why) => panic!("couldn't open {}: {}", display, why),
        Ok(file) => file,
    };
    println!("Processing WhatsApp chat log from {}", display);

    let reader = BufReader::new(file);
    let date_pattern = Regex::new(r"\[\d{4}-\d{2}-\d{2}, \d{2}:\d{2}:\d{2}\]").unwrap();
    let mut messages: Vec<Message> = Vec::new();
    let mut last_date = String::new();
    let mut last_sender = String::new();
    let mut last_content = String::new();

    for line in reader.lines() {
        let line = line.unwrap();
        if line.starts_with("\u{200e}") {
            continue;
        }
        if let Some(date) = date_pattern.find(&line) {
            if !last_content.is_empty() {
                last_content = last_content.trim().to_string();
                let date =
                    chrono::NaiveDateTime::parse_from_str(&last_date, "[%Y-%m-%d, %H:%M:%S]")
                        .unwrap();
                messages.push(Message {
                    date,
                    sender: last_sender.clone(),
                    content: last_content.clone(),
                });
            }
            last_date = date.as_str().to_string();
            last_content = line.replace(&last_date, "").trim().to_string();
            let parts: Vec<&str> = last_content.split(":").collect();
            if parts.len() <= 1 {
                continue;
            }
            last_sender = parts[0].to_string();
            last_content = last_content.replace(&last_sender, "");
            last_content = last_content
                .chars()
                .skip(1)
                .collect::<String>()
                .trim()
                .to_string()
                + " ";

            if last_content.starts_with("\u{200e}") {
                last_date.clear();
                last_sender.clear();
                last_content.clear();
            }
        } else {
            if last_date.is_empty() {
                continue;
            }
            last_content += &line;
        }
    }

    // find number of participants
    let mut participants: HashSet<String> = HashSet::new();
    for message in &messages {
        participants.insert(message.sender.clone());
    }
    let participants: Vec<String> = participants.into_iter().collect();

    println!(
        "Extracted {} messages with {} participants",
        messages.len(),
        participants.len()
    );

    let title = "WhatsApp Chat between ".to_string() + &participants.join(", ");
    // content format is: "DATE;;;;;SENDER;;;;;CONTENT\n"
    let content = messages
        .iter()
        .map(|m| format!("{};;;;;{};;;;;{}", m.date, m.sender, m.content))
        .collect::<Vec<String>>()
        .join("\n");

    let content = insert_content(title.as_str(), content.as_str())
        .await
        .context("Unable to insert content")?;

    for (i, message) in messages.iter().enumerate() {
        print!("Memorizing messages {}/{}\r", i + 1, messages.len());
        // receivers = participants - sender
        let receivers: Vec<String> = participants
            .iter()
            .filter(|p| *p != &message.sender)
            .cloned()
            .collect();

        if message.content.is_empty() {
            continue;
        }

        let res = insert_vector_index(
            content.id.clone(),
            i as u16,
            &message.content,
            json!({
                "sender": message.sender,
                "receivers": receivers,
                "date": message.date.to_string(),
                "source": title.clone(),
            }),
        )
        .await;

        match res {
            Ok(_) => {}
            Err(e) => {
                if e.to_string().contains("Content chunk is empty") {
                    continue;
                }
                println!("Unable to insert vector index: {}", e);
            }
        }
    }
    println!("Memorized {}", title);

    Ok(())
}

pub async fn ingest_via_cli(content: &str) -> anyhow::Result<()> {
    smart_insert_content(
        &format!("Direct insert on {}", Utc::now().date_naive()),
        &content,
        json!({
            "source": "direct insert",
            "time": Utc::now(),
        }),
    )
    .await?;
    Ok(())
}

pub async fn ingest_via_txt_file(path: PathBuf) -> anyhow::Result<()> {
    let display = path.display();
    let file_name = path
        .file_name()
        .context("Unable to get file name")?
        .to_str()
        .context("Unable to convert file name to string")?;
    let file = match File::open(&path) {
        Err(why) => panic!("couldn't open {}: {}", display, why),
        Ok(file) => file,
    };
    println!("Processing text file from {}", display);

    let reader = BufReader::new(file);
    // read all lines and create a single string with "\n" as separator
    let content = reader
        .lines()
        .map(|l| l.unwrap())
        .collect::<Vec<String>>()
        .join("\n");

    let content = smart_insert_content(
        &format!("Contents of {:?}", file_name),
        &content,
        json!({
            "source": file_name,
            "upload_time": Utc::now(),
        }),
    )
    .await?;

    println!("Memorized {}", content.title);

    Ok(())
}

pub async fn ingest_via_pdf_file(path: PathBuf) -> anyhow::Result<()> {
    let display = path.display();
    let bytes = std::fs::read(path.clone()).unwrap();
    let out = pdf_extract::extract_text_from_mem(&bytes).unwrap();

    let file_name = path
        .file_name()
        .context("Unable to get file name")?
        .to_str()
        .context("Unable to convert file name to string")?;

    println!("Processing pdf from {}", display);

    let content = smart_insert_content(
        &format!("Contents of {:?}", file_name),
        &out,
        json!({
            "source": file_name,
            "upload_time": Utc::now(),
        }),
    )
    .await?;

    println!("Memorized {}", content.title);

    Ok(())
}

pub async fn ingest_via_audio_file(path: PathBuf) -> anyhow::Result<()> {
    let display = path.display();
    let file_name = path
        .file_name()
        .context("Unable to get file name")?
        .to_str()
        .context("Unable to convert file name to string")?;

    println!("Processing audio file from {}", display);

    let transcription_points = whisper_decode(path.clone()).await?;

    let transcription = transcription_points
        .iter()
        .map(|t| t.dr.text.clone())
        .collect::<Vec<String>>()
        .join(" ");

    let content = insert_content(file_name, transcription.as_str())
        .await
        .context("Unable to insert content")?;

    for (i, transcription_point) in transcription_points.iter().enumerate() {
        print!(
            "Memorizing messages {}/{}\r",
            i + 1,
            transcription_points.len()
        );

        if transcription_point.dr.text.is_empty() {
            continue;
        }

        let res = insert_vector_index(
            content.id.clone(),
            i as u16,
            &transcription_point.dr.text,
            json!({
                "start_time": transcription_point.start,
                "end_time": transcription_point.start + transcription_point.duration,
                "upload_time": Utc::now(),
                "source": file_name,
            }),
        )
        .await;

        match res {
            Ok(_) => {}
            Err(e) => {
                if e.to_string().contains("Content chunk is empty") {
                    continue;
                }
                println!("Unable to insert vector index: {}", e);
            }
        }
    }
    println!("Memorized {}", file_name);
    Ok(())
}
