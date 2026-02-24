use iced::widget::{button, column, pick_list, row, slider, text, Space};
use iced::Element;

use crate::app::{scaled, Message};
use crate::settings::{BlurShape, Settings};
use crate::theme::muted_color;

pub fn view<'a>(settings: &Settings) -> Element<'a, Message> {
    let fs = settings.font_scale;
    let theme = crate::theme::resolve_theme(settings.appearance, settings.high_contrast);
    let muted = muted_color(&theme);

    column![
        // Blur shape
        text("Detection").size(scaled(16.0, fs)),
        Space::new().height(8),
        row![
            text("Blur shape").size(scaled(13.0, fs)),
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
        .size(scaled(11.0, fs))
        .color(muted),
        Space::new().height(20),
        // Confidence
        text("Tuning").size(scaled(16.0, fs)),
        Space::new().height(8),
        row![
            text("Confidence").size(scaled(13.0, fs)),
            slider(10..=100, settings.confidence, Message::ConfidenceChanged),
            text(format!("{:.2}", settings.confidence as f64 / 100.0)).size(scaled(13.0, fs)),
        ]
        .spacing(12)
        .align_y(iced::Alignment::Center),
        Space::new().height(4),
        text("How certain the detector must be that something is a face.")
            .size(scaled(11.0, fs))
            .color(muted),
        Space::new().height(12),
        // Blur strength
        row![
            text("Blur strength").size(scaled(13.0, fs)),
            slider(
                51..=401,
                settings.blur_strength,
                Message::BlurStrengthChanged
            )
            .step(2u32),
            text(format!("{}", settings.blur_strength)).size(scaled(13.0, fs)),
        ]
        .spacing(12)
        .align_y(iced::Alignment::Center),
        Space::new().height(4),
        text("How heavily faces are blurred.")
            .size(scaled(11.0, fs))
            .color(muted),
        Space::new().height(12),
        // Lookahead
        row![
            text("Lookahead").size(scaled(13.0, fs)),
            slider(0..=30, settings.lookahead, Message::LookaheadChanged),
            text(format!("{}", settings.lookahead)).size(scaled(13.0, fs)),
        ]
        .spacing(12)
        .align_y(iced::Alignment::Center),
        Space::new().height(4),
        text("Future frames to scan ahead. Helps blur faces before they enter the frame.")
            .size(scaled(11.0, fs))
            .color(muted),
        Space::new().height(20),
        // Restore defaults
        button(text("Restore Defaults").size(scaled(13.0, fs)))
            .on_press(Message::RestoreDefaults)
            .padding([8, 16])
            .style(button::secondary),
    ]
    .spacing(0)
    .into()
}
