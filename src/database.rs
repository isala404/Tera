use crate::embeddings::get_embeddings;
use anyhow::{Context, Error, Result};
use async_once::AsyncOnce;
use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use surrealdb::engine::local::{Db, RocksDb};
use surrealdb::sql::{thing, Datetime, Thing, Uuid};
use surrealdb::Surreal;
use tracing::debug;

lazy_static! {
    pub static ref DB: AsyncOnce<Surreal<Db>> = AsyncOnce::new(async {
        let db = connect_db().await.expect("Unable to connect to database");
        db
    });
}

async fn connect_db() -> Result<Surreal<Db>, Box<dyn std::error::Error>> {
    // get directory of current binary
    let path = dirs::config_local_dir()
        .expect("Unable to get local config directory")
        .join("tera").join("database");

    debug!(path = ?path, "Connecting to database");

    let db = Surreal::new::<RocksDb>(path).await?;

    db.use_ns("rag").use_db("content").await?;

    db.query(
        "
            DEFINE TABLE content SCHEMAFULL;

            DEFINE FIELD id ON TABLE content TYPE record;
            DEFINE FIELD title ON TABLE content TYPE string;
            DEFINE FIELD text ON TABLE content TYPE string;
            DEFINE FIELD created_at ON TABLE content TYPE datetime DEFAULT time::now();
            DEFINE INDEX contentIdIndex ON TABLE user COLUMNS id UNIQUE;
        ",
    )
    .await?;

    db.query(
        "
            DEFINE TABLE vector_index SCHEMAFULL;

            DEFINE FIELD id ON TABLE vector_index TYPE record;
            DEFINE FIELD content_id ON TABLE vector_index TYPE record<content>;
            DEFINE FIELD content_chunk ON TABLE vector_index TYPE string;
            DEFINE FIELD chunk_number ON TABLE vector_index TYPE int;
            DEFINE FIELD vector ON TABLE vector_index TYPE array<float>;
            DEFINE FIELD vector.* ON TABLE vector_index TYPE float;
            DEFINE FIELD metadata ON TABLE vector_index FLEXIBLE TYPE object;
            DEFINE FIELD created_at ON TABLE vector_index TYPE datetime DEFAULT time::now();
            DEFINE INDEX vectorIdIndex ON TABLE vector_index COLUMNS id UNIQUE;
        ",
    )
    .await?;

    Ok(db)
}


pub async fn forget_all_content() -> Result<(), Error> {
    let path = dirs::config_local_dir()
        .expect("Unable to get local config directory")
        .join("tera");
    debug!(path = ?path, "Droping database");
    std::fs::remove_dir_all(path)?;

    Ok(())
}


#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Content {
    pub id: Thing,
    pub title: String,
    pub text: String,
    pub created_at: Datetime,
}
impl Content {
    #[allow(dead_code)]
    pub async fn get_vector_indexes(&self) -> Result<Vec<VectorIndex>, Error> {
        let db = DB.get().await.clone();
        let mut result = db
            .query("SELECT * FROM vector_index WHERE content_id = $content")
            .bind(("content", self.id.clone()))
            .await?;
        let vector_indexes: Vec<VectorIndex> = result.take(0)?;

        Ok(vector_indexes)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct VectorIndex {
    pub id: Thing,
    pub content_id: Thing,
    pub content_chunk: String,
    pub chunk_number: u16,
    pub metadata: serde_json::Value,
    pub vector: Vec<f32>,
    pub created_at: Datetime,
}
impl VectorIndex {
    #[allow(dead_code)]
    pub async fn get_content(&self) -> Result<Content, Error> {
        let db = DB.get().await.clone();

        let content: Content = db
            .select(self.content_id.clone())
            .await?
            .context("Unable to get content")?;
        Ok(content)
    }

    pub async fn get_adjacent_chunks(
        &self,
        mut upper: u16,
        lower: u16,
    ) -> Result<Vec<VectorIndex>, Error> {
        let db = DB.get().await.clone();

        if upper > self.chunk_number {
            upper = self.chunk_number;
        }

        let mut result = db
            .query("SELECT * FROM vector_index WHERE content_id = $content AND chunk_number >= $start AND chunk_number <= $end ORDER BY chunk_number ASC")
            .bind(("content", self.content_id.clone()))
            .bind(("start", self.chunk_number - upper))
            .bind(("end", self.chunk_number + lower))
            .await?;
        let vector_indexes: Vec<VectorIndex> = result.take(0)?;

        Ok(vector_indexes)
    }
}

pub async fn insert_content(title: &str, text: &str) -> Result<Content, Error> {
    let db = DB.get().await.clone();
    let id = Uuid::new_v4().0.to_string().replace("-", "");
    let id = thing(format!("content:{}", id).as_str())?;

    let content: Content = db
        .create(("content", id.clone()))
        .content(Content {
            id: id.clone(),
            title: title.to_string(),
            text: text.to_string(),
            created_at: Datetime::default(),
        })
        .await?
        .context("Unable to insert content")?;
    Ok(content)
}

pub async fn insert_vector_index(
    content_id: Thing,
    chunk_number: u16,
    content_chunk: &str,
    metadata: serde_json::Value,
) -> Result<VectorIndex, Error> {
    let db = DB.get().await.clone();
    let id = Uuid::new_v4().0.to_string().replace("-", "");
    let id = thing(format!("vector_index:{}", id).as_str())?;

    let content_chunk = content_chunk
        .chars()
        .filter(|c| c.is_ascii())
        .collect::<String>();

    let content_chunk = content_chunk.trim();

    if content_chunk.is_empty() {
        return Err(anyhow::anyhow!("Content chunk is empty"));
    }

    let vector = get_embeddings(&content_chunk)?.reshape((384,))?.to_vec1()?;

    let vector_index: VectorIndex = db
        .create(("vector_index", id.clone()))
        .content(VectorIndex {
            id: id.clone(),
            content_id,
            chunk_number,
            metadata,
            content_chunk: content_chunk.to_string(),
            vector,
            created_at: Datetime::default(),
        })
        .await?
        .context("Unable to insert vector index")?;

    Ok(vector_index)
}

pub async fn smart_insert_content(
    title: &str,
    text: &str,
    metadata: Value,
) -> Result<Content, Error> {
    let content = insert_content(title, text).await?;

    let mut chunks = text.split("\n").collect::<Vec<&str>>();
    chunks.retain(|c| !c.is_empty());

    // Iterate over the chunks if the chunk is larger than 1000 characters recursively split it and add it to the chunks
    for (i, chunk) in chunks.clone().iter().enumerate() {
        if chunk.len() > 1000 {
            let mut split_chunks = chunk.split(".").collect::<Vec<&str>>();
            split_chunks.retain(|c| !c.is_empty());
            for split_chunk in split_chunks {
                chunks.insert(i, split_chunk);
            }
        }
    }

    for (i, chunk) in chunks.clone().into_iter().enumerate() {
        print!("Memorizing chunk {}/{}\r", i + 1, chunks.len());
        let res = insert_vector_index(content.id.clone(), i as u16, chunk, metadata.clone()).await;
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

    Ok(content)
}

pub async fn get_releted_chunks(query: Vec<f32>) -> Result<Vec<VectorIndex>, Error> {
    let db = DB.get().await.clone();
    let mut result = db
        .query("SELECT *, vector::similarity::cosine(vector, $query) AS score FROM vector_index ORDER BY score DESC LIMIT 4")
        .bind(("query", query))
        .await?;
    let vector_indexes: Vec<VectorIndex> = result.take(0)?;

    Ok(vector_indexes)
}

// get all content ordered by created_at
pub async fn get_all_content(start: u16, limit: u16) -> Result<Vec<Content>, Error> {
    let db = DB.get().await.clone();
    let mut result = db
        .query("SELECT * FROM content ORDER BY created_at DESC LIMIT $limit START $start")
        .bind(("start", start))
        .bind(("limit", limit))
        .await?;
    let content: Vec<Content> = result.take(0)?;

    Ok(content)
}


// Delete content by id
pub async fn delete_content(id: &str) -> Result<(), Error> {
    let db = DB.get().await.clone();
    let id = thing(format!("content:{}", id).as_str())?;

    db.query("DELETE FROM vector_index WHERE content_id = $id")
        .bind(("id", id.clone()))
        .await?.check().context("Unable to delete vector index")?;
    
    db.query("DELETE FROM content WHERE id = $id")
        .bind(("id", id))
        .await?.check().context("Unable to delete content")?;

    Ok(())
}
