// Adopted from https://github.com/huggingface/candle/blob/96f1a28e390fceeaa12b3272c8ac5dcccc8eb5fa/candle-examples/examples/whisper/main.rs

extern crate accelerate_src;
use anyhow::{Context, Error as E, Result};
use candle_core::{Device, IndexOp, Tensor};
use candle_nn::{ops::softmax, VarBuilder};
use candle_transformers::models::whisper::{self as m, audio, Config};
use hf_hub::{api::sync::Api, Repo, RepoType};
use lazy_static::lazy_static;
use rand::{distributions::Distribution, SeedableRng};
use std::path::PathBuf;
use std::process::Command;
use tempfile::tempdir;
use tokenizers::Tokenizer;
use tracing::{debug, error};

lazy_static! {
    pub static ref WHISPER: (m::model::Whisper, Tokenizer, Config) =
        load_model().expect("Unable to load model");
}

const NO_SPEECH_TOKENS: [&str; 2] = ["<|nocaptions|>", "<|nospeech|>"];

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct DecodingResult {
    tokens: Vec<u32>,
    pub text: String,
    avg_logprob: f64,
    no_speech_prob: f64,
    temperature: f64,
    compression_ratio: f64,
}

#[derive(Debug, Clone)]
pub struct Segment {
    pub start: f64,
    pub duration: f64,
    pub dr: DecodingResult,
}

struct Decoder {
    model: m::model::Whisper,
    rng: rand::rngs::StdRng,
    timestamps: bool,
    tokenizer: Tokenizer,
    suppress_tokens: Tensor,
    sot_token: u32,
    transcribe_token: u32,
    eot_token: u32,
    no_speech_token: u32,
    no_timestamps_token: u32,
    language_token: Option<u32>,
}

impl Decoder {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: m::model::Whisper,
        tokenizer: Tokenizer,
        seed: u64,
        device: &Device,
        language_token: Option<u32>,
        timestamps: bool,
    ) -> Result<Self> {
        let no_timestamps_token = token_id(&tokenizer, m::NO_TIMESTAMPS_TOKEN)?;
        // Suppress the notimestamps token when in timestamps mode.
        // https://github.com/openai/whisper/blob/e8622f9afc4eba139bf796c210f5c01081000472/whisper/decoding.py#L452
        let suppress_tokens: Vec<f32> = (0..model.config.vocab_size as u32)
            .map(|i| {
                if model.config.suppress_tokens.contains(&i)
                    || timestamps && i == no_timestamps_token
                {
                    f32::NEG_INFINITY
                } else {
                    0f32
                }
            })
            .collect();
        let suppress_tokens = Tensor::new(suppress_tokens.as_slice(), device)?;
        let sot_token = token_id(&tokenizer, m::SOT_TOKEN)?;
        let transcribe_token = token_id(&tokenizer, m::TRANSCRIBE_TOKEN)?;
        let eot_token = token_id(&tokenizer, m::EOT_TOKEN)?;
        let no_speech_token = NO_SPEECH_TOKENS
            .iter()
            .find_map(|token| token_id(&tokenizer, token).ok());
        let no_speech_token = match no_speech_token {
            None => anyhow::bail!("unable to find any non-speech token"),
            Some(n) => n,
        };
        Ok(Self {
            model,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            tokenizer,
            timestamps,
            suppress_tokens,
            sot_token,
            transcribe_token,
            eot_token,
            no_speech_token,
            language_token,
            no_timestamps_token,
        })
    }

    fn decode(&mut self, mel: &Tensor, t: f64) -> Result<DecodingResult> {
        let model = &mut self.model;
        let audio_features = model.encoder.forward(mel, true)?;
        let sample_len = model.config.max_target_positions / 2;
        let mut sum_logprob = 0f64;
        let mut no_speech_prob = f64::NAN;
        let mut tokens = vec![self.sot_token];
        if let Some(language_token) = self.language_token {
            tokens.push(language_token);
        }
        tokens.push(self.transcribe_token);
        if !self.timestamps {
            tokens.push(self.no_timestamps_token);
        }
        for i in 0..sample_len {
            let tokens_t = Tensor::new(tokens.as_slice(), mel.device())?;

            // The model expects a batch dim but this inference loop does not handle
            // it so we add it at this point.
            let tokens_t = tokens_t.unsqueeze(0)?;
            let ys = model.decoder.forward(&tokens_t, &audio_features, i == 0)?;

            // Extract the no speech probability on the first iteration by looking at the first
            // token logits and the probability for the according token.
            if i == 0 {
                let logits = model.decoder.final_linear(&ys.i(..1)?)?.i(0)?.i(0)?;
                no_speech_prob = softmax(&logits, 0)?
                    .i(self.no_speech_token as usize)?
                    .to_scalar::<f32>()? as f64;
            }

            let (_, seq_len, _) = ys.dims3()?;
            let logits = model
                .decoder
                .final_linear(&ys.i((..1, seq_len - 1..))?)?
                .i(0)?
                .i(0)?;
            // TODO: Besides suppress tokens, we should apply the heuristics from
            // ApplyTimestampRules, i.e.:
            // - Timestamps come in pairs, except before EOT.
            // - Timestamps should be non-decreasing.
            // - If the sum of the probabilities of timestamps is higher than any other tokens,
            //   only consider timestamps when sampling.
            // https://github.com/openai/whisper/blob/e8622f9afc4eba139bf796c210f5c01081000472/whisper/decoding.py#L439
            let logits = logits.broadcast_add(&self.suppress_tokens)?;
            let next_token = if t > 0f64 {
                let prs = softmax(&(&logits / t)?, 0)?;
                let logits_v: Vec<f32> = prs.to_vec1()?;
                let distr = rand::distributions::WeightedIndex::new(&logits_v)?;
                distr.sample(&mut self.rng) as u32
            } else {
                let logits_v: Vec<f32> = logits.to_vec1()?;
                logits_v
                    .iter()
                    .enumerate()
                    .max_by(|(_, u), (_, v)| u.total_cmp(v))
                    .map(|(i, _)| i as u32)
                    .unwrap()
            };
            tokens.push(next_token);
            let prob = softmax(&logits, candle_core::D::Minus1)?
                .i(next_token as usize)?
                .to_scalar::<f32>()? as f64;
            if next_token == self.eot_token || tokens.len() > model.config.max_target_positions {
                break;
            }
            sum_logprob += prob.ln();
        }
        let text = self.tokenizer.decode(&tokens, true).map_err(E::msg)?;
        let avg_logprob = sum_logprob / tokens.len() as f64;

        Ok(DecodingResult {
            tokens,
            text,
            avg_logprob,
            no_speech_prob,
            temperature: t,
            compression_ratio: f64::NAN,
        })
    }

    fn decode_with_fallback(&mut self, segment: &Tensor) -> Result<DecodingResult> {
        for (i, &t) in m::TEMPERATURES.iter().enumerate() {
            let dr: Result<DecodingResult> = self.decode(segment, t);
            if i == m::TEMPERATURES.len() - 1 {
                return dr;
            }
            // On errors, we try again with a different temperature.
            match dr {
                Ok(dr) => {
                    let needs_fallback = dr.compression_ratio > m::COMPRESSION_RATIO_THRESHOLD
                        || dr.avg_logprob < m::LOGPROB_THRESHOLD;
                    if !needs_fallback || dr.no_speech_prob > m::NO_SPEECH_THRESHOLD {
                        return Ok(dr);
                    }
                }
                Err(err) => {
                    println!("Error running at {t}: {err}")
                }
            }
        }
        unreachable!()
    }

    fn run(&mut self, mel: &Tensor) -> Result<Vec<Segment>> {
        let (_, _, content_frames) = mel.dims3()?;
        let mut seek = 0;
        let mut segments = vec![];
        while seek < content_frames {
            let time_offset = (seek * m::HOP_LENGTH) as f64 / m::SAMPLE_RATE as f64;
            let segment_size = usize::min(content_frames - seek, m::N_FRAMES);
            let mel_segment = mel.narrow(2, seek, segment_size)?;
            let segment_duration = (segment_size * m::HOP_LENGTH) as f64 / m::SAMPLE_RATE as f64;
            let dr = self.decode_with_fallback(&mel_segment)?;
            seek += segment_size;
            if dr.no_speech_prob > m::NO_SPEECH_THRESHOLD && dr.avg_logprob < m::LOGPROB_THRESHOLD {
                debug!("no speech detected, skipping {seek} {dr:?}");
                continue;
            }
            let segment = Segment {
                start: time_offset,
                duration: segment_duration,
                dr,
            };
            segments.push(segment)
        }
        Ok(segments)
    }
}

pub fn token_id(tokenizer: &Tokenizer, token: &str) -> candle_core::Result<u32> {
    match tokenizer.token_to_id(token) {
        None => candle_core::bail!("no token-id for {token}"),
        Some(id) => Ok(id),
    }
}

pub async fn whisper_decode(path: PathBuf) -> anyhow::Result<Vec<Segment>> {
    // let input = std::path::PathBuf::from(path);
    let (model, tokenizer, config) = &*WHISPER;
    let device = Device::Cpu;

    let temp_dir = match tempdir() {
        Ok(temp_dir) => temp_dir,
        Err(error) => {
            error!("Unable to create temporary directory: {}", error);
            return Err(anyhow::anyhow!("Unable to create temporary directory"));
        }
    };

    let dir_path = temp_dir.path();

    resample(&path, dir_path).context("Unable to resample file")?;

    let mel_bytes = include_bytes!("melfilters.bytes").as_slice();

    let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
    <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(mel_bytes, &mut mel_filters);

    let mut input = std::fs::File::open(dir_path.join("tempfile.wav"))?;
    let (header, data) = wav::read(&mut input)?;

    if header.sampling_rate != m::SAMPLE_RATE as u32 {
        anyhow::bail!("wav file must have a {} sampling rate", m::SAMPLE_RATE)
    }
    let data = data.as_sixteen().expect("expected 16 bit wav file");
    let pcm_data: Vec<_> = data[..data.len() / header.channel_count as usize]
        .iter()
        .map(|v| *v as f32 / 32768.)
        .collect();
    let mel = audio::pcm_to_mel(&config, &pcm_data, &mel_filters);
    let mel_len = mel.len();
    let mel = Tensor::from_vec(
        mel,
        (1, config.num_mel_bins, mel_len / config.num_mel_bins),
        &device,
    )?;

    let language_token = None;
    let mut dc = Decoder::new(
        model.clone(),
        tokenizer.clone(),
        299792458,
        &device,
        language_token,
        false,
    )?;
    let result = dc.run(&mel)?;
    Ok(result)
}

pub fn load_model() -> Result<(m::model::Whisper, Tokenizer, Config)> {
    let api = Api::new()?.repo(Repo::with_revision(
        "distil-whisper/distil-small.en".to_string(),
        RepoType::Model,
        "ca96ef6945f9c4cedee45d33833bff2c65f960d4".to_string(),
    ));
    let config_filename = api.get("config.json")?;
    let tokenizer_filename = api.get("tokenizer.json")?;
    let weights_filename = api.get("pytorch_model.bin")?;

    let config = std::fs::read_to_string(config_filename)?;
    let config: Config = serde_json::from_str(&config)?;

    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let vb = VarBuilder::from_pth(weights_filename, m::DTYPE, &Device::Cpu)?;
    let model = m::model::Whisper::load(&vb, config.clone())?;

    Ok((model, tokenizer, config))
}

fn resample(in_file: &PathBuf, dir_path: &std::path::Path) -> anyhow::Result<()> {
    Command::new("ffmpeg")
        .arg("-i")
        .arg(in_file)
        .arg("-ac")
        .arg("1")
        .arg("-ar")
        .arg("16000")
        .arg(dir_path.join("tempfile.wav"))
        .output()
        .context("Unable to resample file")?;
    Ok(())
}
