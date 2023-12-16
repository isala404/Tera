// Adopted from https://github.com/huggingface/candle/blob/96f1a28e390fceeaa12b3272c8ac5dcccc8eb5fa/candle-examples/examples/phi/main.rs

extern crate accelerate_src;
use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_mixformer::Config;
use candle_transformers::models::quantized_mixformer::MixFormerSequentialForCausalLM as QMixFormer;
use hf_hub::{api::sync::Api, Repo};
use lazy_static::lazy_static;
use serde_json::json;
use tokenizers::Tokenizer;
use tracing::debug;

use crate::database::VectorIndex;

lazy_static! {
    pub static ref PHI: (QMixFormer, Tokenizer) = load_model().expect("Unable to load model");
}

pub fn load_model() -> Result<(QMixFormer, Tokenizer)> {
    let api = Api::new()?.repo(Repo::model("lmz/candle-quantized-phi".to_string()));
    let tokenizer_filename = api.get("tokenizer.json")?;
    let weights_filename = api.get("model-v2-q4k.gguf")?;

    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
    let config = Config::v2();
    let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(&weights_filename)?;
    let model = QMixFormer::new_v2(&config, vb)?;

    Ok((model, tokenizer))
}

struct TextGeneration {
    model: QMixFormer,
    device: Device,
    tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: QMixFormer,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: &Device,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Self {
            model,
            tokenizer,
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
        }
    }

    fn run(&mut self, prompt: &str, sample_len: usize) -> Result<String> {
        debug!(prompt = prompt, "starting the inference loop");
        let tokens = self.tokenizer.encode(prompt, true).map_err(E::msg)?;
        if tokens.is_empty() {
            anyhow::bail!("Empty prompts are not supported in the phi model.")
        }
        let mut tokens = tokens.get_ids().to_vec();
        let mut generated_tokens = 0usize;
        let eos_token = match self.tokenizer.get_vocab(true).get("<|endoftext|>") {
            Some(token) => *token,
            None => anyhow::bail!("cannot find the endoftext token"),
        };
        let start_gen = std::time::Instant::now();

        let mut response = String::new();

        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input)?;
            let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )?
            };

            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;
            if next_token == eos_token || next_token == 198 {
                break;
            }
            let token = self.tokenizer.decode(&[next_token], true).map_err(E::msg)?;
            response += &token;
        }
        let dt = start_gen.elapsed();
        debug!(
            generated_tokens = generated_tokens,
            speed = format!("{:.2} token/s", generated_tokens as f64 / dt.as_secs_f64()),
            "inference loop finished"
        );
        Ok(response.trim().to_string())
    }
}

pub async fn answer_with_context(query: &str, references: Vec<VectorIndex>) -> Result<String> {
    if references.is_empty() {
        return Ok("Non of your saved content is relevant to this question. I can only answer based on your saved content.".to_string());
    }

    let mut context = Vec::new();
    for reference in references.clone() {
        context.push(json!(
            {
                "content": reference.content_chunk,
                "metadata": reference.metadata,
            }
        ))
    }

    let context = json!(context).to_string();

    let prompt = format!("Tera: Your task as an AI assistant is to generate the answer to the user's question \"{question}\" by analyzing both the JSON context: \"{context}\" and any available metadata. If the context or metadata lacks information to answer the question, indicate the absence of relevant details.\nAnswer:", context=context, question=query);

    debug!(prompt =? prompt, "Synthesizing answer with context");

    let (model, tokenizer) = &*PHI;
    let device = Device::Cpu;

    let mut pipeline = TextGeneration::new(
        model.clone(),
        tokenizer.clone(),
        299792458,
        Some(0.0),
        None,
        1.1,
        64,
        &device,
    );
    let response = pipeline.run(&prompt, 400)?;

    Ok(response)
}
