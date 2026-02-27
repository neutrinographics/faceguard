use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BlurShape {
    Ellipse,
    Rect,
}

impl BlurShape {
    pub const ALL: &[BlurShape] = &[BlurShape::Ellipse, BlurShape::Rect];
}

impl std::fmt::Display for BlurShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BlurShape::Ellipse => write!(f, "Ellipse"),
            BlurShape::Rect => write!(f, "Rectangle"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Appearance {
    System,
    Dark,
    Light,
}

impl Appearance {
    pub const ALL: &[Appearance] = &[Appearance::System, Appearance::Dark, Appearance::Light];
}

impl std::fmt::Display for Appearance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Appearance::System => write!(f, "System"),
            Appearance::Dark => write!(f, "Dark"),
            Appearance::Light => write!(f, "Light"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VoiceDisguise {
    Off,
    Low,
    Medium,
    High,
}

impl VoiceDisguise {
    pub const ALL: &[VoiceDisguise] = &[
        VoiceDisguise::Off,
        VoiceDisguise::Low,
        VoiceDisguise::Medium,
        VoiceDisguise::High,
    ];
}

impl std::fmt::Display for VoiceDisguise {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VoiceDisguise::Off => write!(f, "Off"),
            VoiceDisguise::Low => write!(f, "Low"),
            VoiceDisguise::Medium => write!(f, "Medium"),
            VoiceDisguise::High => write!(f, "High"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BleepSound {
    Tone,
    Silence,
}

impl BleepSound {
    pub const ALL: &[BleepSound] = &[BleepSound::Tone, BleepSound::Silence];
}

impl std::fmt::Display for BleepSound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BleepSound::Tone => write!(f, "Tone"),
            BleepSound::Silence => write!(f, "Silence"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    pub blur_shape: BlurShape,
    pub confidence: u32,
    pub blur_strength: u32,
    #[serde(default = "default_blur_coverage")]
    pub blur_coverage: u32,
    #[serde(default)]
    pub center_offset: i32,
    pub lookahead: u32,
    #[serde(default = "default_quality")]
    pub quality: u32,
    pub appearance: Appearance,
    pub high_contrast: bool,
    pub font_scale: f32,
    #[serde(default)]
    pub audio_processing: bool,
    #[serde(default)]
    pub bleep_keywords: String,
    #[serde(default = "default_bleep_sound")]
    pub bleep_sound: BleepSound,
    #[serde(default = "default_voice_disguise")]
    pub voice_disguise: VoiceDisguise,
}

fn default_blur_coverage() -> u32 {
    40
}

fn default_bleep_sound() -> BleepSound {
    BleepSound::Tone
}

fn default_voice_disguise() -> VoiceDisguise {
    VoiceDisguise::Off
}

fn default_quality() -> u32 {
    crf_to_quality(faceguard_core::video::infrastructure::ffmpeg_writer::DEFAULT_CRF)
}

pub fn quality_to_crf(quality: u32) -> u32 {
    51 - (quality.min(100) * 51 / 100)
}

fn crf_to_quality(crf: u32) -> u32 {
    (51 - crf.min(51)) * 100 / 51
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            blur_shape: BlurShape::Ellipse,
            confidence: 50,
            blur_strength: 201,
            blur_coverage: 40,
            center_offset: 0,
            lookahead: 10,
            quality: default_quality(),
            appearance: Appearance::System,
            high_contrast: false,
            font_scale: 1.0,
            audio_processing: false,
            bleep_keywords: String::new(),
            bleep_sound: default_bleep_sound(),
            voice_disguise: default_voice_disguise(),
        }
    }
}

impl Settings {
    fn config_path() -> Option<PathBuf> {
        dirs::config_dir().map(|d| d.join("FaceGuard").join("settings.json"))
    }

    pub fn load() -> Self {
        Self::config_path()
            .and_then(|path| fs::read_to_string(path).ok())
            .and_then(|json| serde_json::from_str(&json).ok())
            .unwrap_or_default()
    }

    pub fn save(&self) {
        if let Some(path) = Self::config_path() {
            if let Some(parent) = path.parent() {
                let _ = fs::create_dir_all(parent);
            }
            if let Ok(json) = serde_json::to_string_pretty(self) {
                let _ = fs::write(path, json);
            }
        }
    }
}
