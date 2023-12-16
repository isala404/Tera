use anyhow::{Context, Error as E, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use hf_hub::{api::sync::Api, Repo};
use lazy_static::lazy_static;
use tokenizers::{PaddingParams, Tokenizer};

lazy_static! {
    pub static ref AI: (BertModel, Tokenizer) = load_model().expect("Unable to load model");
}

pub fn load_model() -> Result<(BertModel, Tokenizer)> {
    let api = Api::new()?.repo(Repo::model("BAAI/bge-small-en-v1.5".to_string()));
    let config_filename = api.get("config.json")?;
    let tokenizer_filename = api.get("tokenizer.json")?;
    let weights_filename = api.get("pytorch_model.bin")?;

    let config = std::fs::read_to_string(config_filename)?;
    let config: Config = serde_json::from_str(&config)?;

    let mut tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let vb = VarBuilder::from_pth(&weights_filename, DTYPE, &Device::Cpu)?;
    let model = BertModel::load(vb, &config)?;

    if let Some(pp) = tokenizer.get_padding_mut() {
        pp.strategy = tokenizers::PaddingStrategy::BatchLongest
    } else {
        let pp = PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        };
        tokenizer.with_padding(Some(pp));
    }

    Ok((model, tokenizer))
}

pub fn get_embeddings(sentence: &str) -> Result<Tensor> {
    let (model, tokenizer) = &*AI;

    // drop any non-ascii characters
    let sentence = sentence
        .chars()
        .filter(|c| c.is_ascii())
        .collect::<String>();

    let tokens = tokenizer
        .encode_batch(vec![sentence], true)
        .map_err(E::msg)
        .context("Unable to encode sentence")?;

    let token_ids = tokens
        .iter()
        .map(|tokens| {
            let tokens = tokens.get_ids().to_vec();
            Ok(Tensor::new(tokens.as_slice(), &Device::Cpu)?)
        })
        .collect::<Result<Vec<_>>>()
        .context("Unable to get token ids")?;

    let token_ids = Tensor::stack(&token_ids, 0).context("Unable to stack token ids")?;
    let token_type_ids = token_ids
        .zeros_like()
        .context("Unable to get token type ids")?;

    let embeddings = model
        .forward(&token_ids, &token_type_ids)
        .context("Unable to get embeddings")?;

    let (_n_sentence, n_tokens, _hidden_size) = embeddings
        .dims3()
        .context("Unable to get embeddings dimensions")?;
    let embeddings =
        (embeddings.sum(1)? / (n_tokens as f64)).context("Unable to get embeddings sum")?;
    let embeddings = embeddings
        .broadcast_div(&embeddings.sqr()?.sum_keepdim(1)?.sqrt()?)
        .context("Unable to get embeddings broadcast div")?;

    Ok(embeddings)
}
