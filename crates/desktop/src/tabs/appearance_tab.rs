use iced::widget::{checkbox, column, pick_list, row, slider, text, Space};
use iced::Element;

use crate::app::{scaled, Message};
use crate::settings::{Appearance, Settings};

pub fn view<'a>(settings: &Settings) -> Element<'a, Message> {
    let fs = settings.font_scale;

    column![
        text("Theme").size(scaled(16.0, fs)),
        Space::new().height(8),
        row![
            text("Mode").size(scaled(13.0, fs)),
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
            .text_size(scaled(13.0, fs)),
        Space::new().height(20),
        text("Font size").size(scaled(16.0, fs)),
        Space::new().height(8),
        row![
            slider(0.8..=1.5, settings.font_scale, Message::FontScaleChanged).step(0.05),
            text(format!("{:.0}%", settings.font_scale * 100.0)).size(scaled(13.0, fs)),
        ]
        .spacing(12)
        .align_y(iced::Alignment::Center),
    ]
    .spacing(0)
    .into()
}
