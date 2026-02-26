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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    pub blur_shape: BlurShape,
    pub confidence: u32,
    pub blur_strength: u32,
    pub lookahead: u32,
    #[serde(default = "default_quality")]
    pub quality: u32,
    pub appearance: Appearance,
    pub high_contrast: bool,
    pub font_scale: f32,
}

fn default_quality() -> u32 {
    crf_to_quality(video_blur_core::video::infrastructure::ffmpeg_writer::DEFAULT_CRF)
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
            lookahead: 10,
            quality: default_quality(),
            appearance: Appearance::System,
            high_contrast: false,
            font_scale: 1.0,
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
