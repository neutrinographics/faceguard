use std::path::Path;

use iced::widget::{button, column, row, text, Space};
use iced::{Element, Length};

use crate::app::{scaled, Message};

pub fn view<'a>(
    fs: f32,
    input_path: Option<&Path>,
    output_path: Option<&Path>,
) -> Element<'a, Message> {
    let mut col = column![
        text("Blur faces in videos and photos. Select a file to get started.")
            .size(scaled(13.0, fs)),
        Space::new().height(12),
        // Input file row
        file_row(fs, "Input", input_path, Message::SelectInput),
        Space::new().height(8),
        // Output file row
        file_row_maybe(
            fs,
            "Output",
            output_path,
            Message::SelectOutput,
            input_path.is_some()
        ),
    ]
    .spacing(0);

    if input_path.is_some() {
        // Controls panel will be populated in later phases (7E/7F)
        col = col.push(Space::new().height(16)).push(
            text("Ready to process. Run and face selection coming soon.").size(scaled(13.0, fs)),
        );
    }

    col.into()
}

/// A file row: label + filename + Browse button.
fn file_row(
    fs: f32,
    label: &str,
    path: Option<&Path>,
    on_browse: Message,
) -> Element<'static, Message> {
    let display = path
        .and_then(|p| p.file_name())
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "No file selected".to_string());

    row![
        column![
            text(label.to_string()).size(scaled(11.0, fs)),
            text(display).size(scaled(13.0, fs)),
        ]
        .width(Length::Fill),
        button(text("Browse").size(scaled(13.0, fs)))
            .on_press(on_browse)
            .padding([6, 16]),
    ]
    .spacing(8)
    .align_y(iced::Alignment::Center)
    .into()
}

/// A file row that can be disabled.
fn file_row_maybe(
    fs: f32,
    label: &str,
    path: Option<&Path>,
    on_browse: Message,
    enabled: bool,
) -> Element<'static, Message> {
    let display = path
        .and_then(|p| p.file_name())
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "No file selected".to_string());

    let btn = button(text("Browse").size(scaled(13.0, fs))).padding([6, 16]);
    let btn = if enabled {
        btn.on_press(on_browse)
    } else {
        btn
    };

    row![
        column![
            text(label.to_string()).size(scaled(11.0, fs)),
            text(display).size(scaled(13.0, fs)),
        ]
        .width(Length::Fill),
        btn,
    ]
    .spacing(8)
    .align_y(iced::Alignment::Center)
    .into()
}
