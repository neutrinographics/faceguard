use std::path::Path;

use iced::widget::{button, column, progress_bar, row, text, Space};
use iced::{Element, Length};

use crate::app::{scaled, Message, ProcessingState};

pub fn view<'a>(
    fs: f32,
    input_path: Option<&Path>,
    output_path: Option<&Path>,
    processing: &ProcessingState,
) -> Element<'a, Message> {
    let is_processing = matches!(
        processing,
        ProcessingState::Preparing | ProcessingState::Downloading(..) | ProcessingState::Blurring(..)
    );

    let mut col = column![
        text("Blur faces in videos and photos. Select a file to get started.")
            .size(scaled(13.0, fs)),
        Space::new().height(12),
        file_row(
            fs,
            "Input",
            input_path,
            Message::SelectInput,
            !is_processing
        ),
        Space::new().height(8),
        file_row(
            fs,
            "Output",
            output_path,
            Message::SelectOutput,
            input_path.is_some() && !is_processing,
        ),
    ]
    .spacing(0);

    if input_path.is_some() {
        col = col.push(Space::new().height(16));

        match processing {
            ProcessingState::Idle => {
                col = col.push(
                    button(text("Run").size(scaled(13.0, fs)))
                        .on_press(Message::RunBlur)
                        .padding([8, 24]),
                );
            }
            ProcessingState::Preparing => {
                col = col
                    .push(text("Loading model...").size(scaled(13.0, fs)))
                    .push(Space::new().height(8))
                    .push(
                        button(text("Cancel").size(scaled(13.0, fs)))
                            .on_press(Message::CancelBlur)
                            .padding([6, 16]),
                    );
            }
            ProcessingState::Downloading(downloaded, total) => {
                let status = if *total > 0 {
                    let pct = (*downloaded as f64 / *total as f64 * 100.0) as u32;
                    format!("Downloading model... {pct}%")
                } else {
                    format!("Downloading model... {} bytes", downloaded)
                };
                col = col
                    .push(text(status).size(scaled(13.0, fs)))
                    .push(Space::new().height(8))
                    .push(
                        button(text("Cancel").size(scaled(13.0, fs)))
                            .on_press(Message::CancelBlur)
                            .padding([6, 16]),
                    );
            }
            ProcessingState::Blurring(current, total) => {
                let pct = if *total > 0 {
                    *current as f32 / *total as f32 * 100.0
                } else {
                    0.0
                };
                let status = if *total > 0 {
                    format!("Processing frame {current}/{total}")
                } else {
                    format!("Processing frame {current}...")
                };
                col = col
                    .push(text(status).size(scaled(13.0, fs)))
                    .push(Space::new().height(8))
                    .push(progress_bar(0.0..=100.0, pct))
                    .push(Space::new().height(8))
                    .push(
                        button(text("Cancel").size(scaled(13.0, fs)))
                            .on_press(Message::CancelBlur)
                            .padding([6, 16]),
                    );
            }
            ProcessingState::Complete => {
                col = col
                    .push(text("Processing complete!").size(scaled(14.0, fs)))
                    .push(Space::new().height(8))
                    .push(
                        button(text("Run Again").size(scaled(13.0, fs)))
                            .on_press(Message::RunBlur)
                            .padding([8, 24]),
                    );
            }
            ProcessingState::Error(e) => {
                col = col
                    .push(text(format!("Error: {e}")).size(scaled(13.0, fs)))
                    .push(Space::new().height(8))
                    .push(
                        button(text("Retry").size(scaled(13.0, fs)))
                            .on_press(Message::RunBlur)
                            .padding([8, 24]),
                    );
            }
        }
    }

    col.into()
}

fn file_row(
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
