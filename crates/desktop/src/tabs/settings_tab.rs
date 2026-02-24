use iced::widget::{button, checkbox, column, pick_list, row, rule, slider, text, Space};
use iced::Element;

use crate::app::{scaled, Message};
use crate::settings::{Appearance, BlurShape, Settings};
use crate::theme::{muted_color, section_color};

pub fn view<'a>(settings: &Settings, gpu_available: bool) -> Element<'a, Message> {
    let fs = settings.font_scale;
    let theme = crate::theme::resolve_theme(settings.appearance, settings.high_contrast);
    let muted = muted_color(&theme);
    let section = section_color(&theme);

    // --- BLUR SETTINGS ---
    let blur_intensity_label = blur_intensity_label(settings.blur_strength);
    let blur_section = column![
        text("BLUR SETTINGS").size(scaled(12.0, fs)).color(section),
        Space::new().height(12),
        row![
            text("Blur shape").size(scaled(14.0, fs)),
            pick_list(
                BlurShape::ALL,
                Some(settings.blur_shape),
                Message::BlurShapeChanged
            )
            .text_size(scaled(13.0, fs)),
        ]
        .spacing(12)
        .align_y(iced::Alignment::Center),
        Space::new().height(4),
        text(match settings.blur_shape {
            BlurShape::Ellipse => "Ellipse follows the natural shape of a face.",
            BlurShape::Rect => "Rectangle covers a wider area.",
        })
        .size(scaled(12.0, fs))
        .color(muted),
        Space::new().height(16),
        row![
            text("Blur intensity").size(scaled(14.0, fs)),
            slider(
                51..=401,
                settings.blur_strength,
                Message::BlurStrengthChanged
            )
            .step(2u32),
            text(blur_intensity_label).size(scaled(13.0, fs)),
        ]
        .spacing(12)
        .align_y(iced::Alignment::Center),
        Space::new().height(4),
        text("How heavily faces are blurred.")
            .size(scaled(12.0, fs))
            .color(muted),
        Space::new().height(8),
        text(if gpu_available {
            "Blur backend: GPU accelerated"
        } else {
            "Blur backend: CPU"
        })
        .size(scaled(11.0, fs))
        .color(muted),
    ]
    .spacing(0);

    // --- DETECTION ---
    let sensitivity_label = sensitivity_label(settings.confidence);
    let detection_section = column![
        text("DETECTION").size(scaled(12.0, fs)).color(section),
        Space::new().height(12),
        row![
            text("Sensitivity").size(scaled(14.0, fs)),
            slider(10..=100, settings.confidence, Message::ConfidenceChanged),
            text(sensitivity_label).size(scaled(13.0, fs)),
        ]
        .spacing(12)
        .align_y(iced::Alignment::Center),
        Space::new().height(4),
        text("How certain the detector must be that something is a face.")
            .size(scaled(12.0, fs))
            .color(muted),
        Space::new().height(16),
        row![
            text("Lookahead frames").size(scaled(14.0, fs)),
            slider(0..=30, settings.lookahead, Message::LookaheadChanged),
            text(format!("{}", settings.lookahead)).size(scaled(13.0, fs)),
        ]
        .spacing(12)
        .align_y(iced::Alignment::Center),
        Space::new().height(4),
        text("Future frames to scan ahead. Helps blur faces before they enter the frame.")
            .size(scaled(12.0, fs))
            .color(muted),
    ]
    .spacing(0);

    // --- APPEARANCE ---
    let appearance_section = column![
        text("APPEARANCE").size(scaled(12.0, fs)).color(section),
        Space::new().height(12),
        row![
            text("Theme").size(scaled(14.0, fs)),
            pick_list(Appearance::ALL, Some(settings.appearance), |a| {
                Message::AppearanceChanged(a)
            })
            .text_size(scaled(13.0, fs)),
        ]
        .spacing(12)
        .align_y(iced::Alignment::Center),
        Space::new().height(12),
        checkbox(settings.high_contrast)
            .label("High contrast")
            .on_toggle(Message::HighContrastChanged)
            .text_size(scaled(14.0, fs)),
        Space::new().height(16),
        row![
            text("Interface scale").size(scaled(14.0, fs)),
            slider(0.8..=1.5, settings.font_scale, Message::FontScaleChanged).step(0.05),
            text(format!("{:.0}%", settings.font_scale * 100.0)).size(scaled(13.0, fs)),
        ]
        .spacing(12)
        .align_y(iced::Alignment::Center),
    ]
    .spacing(0);

    column![
        blur_section,
        Space::new().height(20),
        rule::horizontal(1),
        Space::new().height(20),
        detection_section,
        Space::new().height(20),
        rule::horizontal(1),
        Space::new().height(20),
        appearance_section,
        Space::new().height(24),
        button(text("Restore Defaults").size(scaled(13.0, fs)))
            .on_press(Message::RestoreDefaults)
            .padding([8, 16])
            .style(button::secondary),
    ]
    .spacing(0)
    .into()
}

fn blur_intensity_label(strength: u32) -> String {
    let qual = match strength {
        51..=150 => "Light",
        151..=275 => "Medium",
        _ => "Heavy",
    };
    format!("{qual} ({strength})")
}

fn sensitivity_label(confidence: u32) -> String {
    let qual = match confidence {
        10..=35 => "Low",
        36..=65 => "Medium",
        _ => "High",
    };
    format!("{qual} ({:.2})", confidence as f64 / 100.0)
}
