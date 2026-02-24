use std::path::Path;

use iced::widget::{button, column, container, progress_bar, row, text, Space};
use iced::{Element, Length, Theme};

use crate::app::{scaled, Message, ProcessingState};
use crate::theme::muted_color;
use crate::widgets::faces_well::{self, FacesWellState};

pub fn view<'a>(
    fs: f32,
    input_path: Option<&Path>,
    output_path: Option<&Path>,
    processing: &ProcessingState,
    faces_well: &FacesWellState,
    theme: &Theme,
) -> Element<'a, Message> {
    let is_processing = matches!(
        processing,
        ProcessingState::Preparing
            | ProcessingState::Downloading(..)
            | ProcessingState::Scanning(..)
            | ProcessingState::Blurring(..)
    );

    let muted = muted_color(theme);

    let mut col = column![
        text("Blur faces in videos and photos. Select a file to get started.")
            .size(scaled(13.0, fs))
            .color(muted),
        Space::new().height(12),
        file_row(
            fs,
            "Input",
            input_path,
            Message::SelectInput,
            !is_processing,
            theme,
        ),
        Space::new().height(6),
        file_row(
            fs,
            "Output",
            output_path,
            Message::SelectOutput,
            input_path.is_some() && !is_processing,
            theme,
        ),
    ]
    .spacing(0);

    if input_path.is_some() {
        col = col.push(Space::new().height(16));

        match processing {
            ProcessingState::Idle => {
                col = col.push(
                    button(text("Run").size(scaled(14.0, fs)))
                        .on_press(Message::RunBlur)
                        .padding([10, 24]),
                );

                col = col.push(Space::new().height(8));
                col = col.push(
                    button(text("Choose which faces to blur...").size(scaled(13.0, fs)))
                        .on_press(Message::RunPreview)
                        .padding([8, 24])
                        .style(button::secondary),
                );
            }
            ProcessingState::Preparing => {
                col = col
                    .push(text("Loading model...").size(scaled(13.0, fs)).color(muted))
                    .push(Space::new().height(8))
                    .push(
                        button(text("Cancel").size(scaled(13.0, fs)))
                            .on_press(Message::CancelWork)
                            .padding([6, 16])
                            .style(button::secondary),
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
                    .push(text(status).size(scaled(13.0, fs)).color(muted))
                    .push(Space::new().height(8))
                    .push(
                        button(text("Cancel").size(scaled(13.0, fs)))
                            .on_press(Message::CancelWork)
                            .padding([6, 16])
                            .style(button::secondary),
                    );
            }
            ProcessingState::Scanning(current, total) => {
                let pct = if *total > 0 {
                    *current as f32 / *total as f32 * 100.0
                } else {
                    0.0
                };
                let status = if *total > 0 {
                    format!("Scanning frame {current}/{total}")
                } else {
                    format!("Scanning frame {current}...")
                };
                col = col
                    .push(text(status).size(scaled(13.0, fs)).color(muted))
                    .push(Space::new().height(8))
                    .push(progress_bar(0.0..=100.0, pct))
                    .push(Space::new().height(8))
                    .push(
                        button(text("Cancel").size(scaled(13.0, fs)))
                            .on_press(Message::CancelWork)
                            .padding([6, 16])
                            .style(button::secondary),
                    );
            }
            ProcessingState::Previewed => {
                col = col.push(faces_well::view(faces_well, fs, theme));
                col = col.push(Space::new().height(12));
                col = col.push(
                    row![
                        button(text("Run").size(scaled(14.0, fs)))
                            .on_press(Message::RunBlur)
                            .padding([10, 24]),
                        button(text("Re-scan").size(scaled(13.0, fs)))
                            .on_press(Message::RunPreview)
                            .padding([8, 24])
                            .style(button::secondary),
                    ]
                    .spacing(8),
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
                    .push(text(status).size(scaled(13.0, fs)).color(muted))
                    .push(Space::new().height(8))
                    .push(progress_bar(0.0..=100.0, pct))
                    .push(Space::new().height(8))
                    .push(
                        button(text("Cancel").size(scaled(13.0, fs)))
                            .on_press(Message::CancelWork)
                            .padding([6, 16])
                            .style(button::secondary),
                    );
            }
            ProcessingState::Complete => {
                col = col
                    .push(text("Processing complete!").size(scaled(14.0, fs)))
                    .push(Space::new().height(12))
                    .push(
                        row![
                            button(text("OK").size(scaled(13.0, fs)))
                                .on_press(Message::DismissComplete)
                                .padding([8, 24]),
                            button(text("Show in Folder").size(scaled(13.0, fs)))
                                .on_press(Message::ShowInFolder)
                                .padding([8, 24])
                                .style(button::secondary),
                        ]
                        .spacing(8),
                    );
            }
            ProcessingState::Error(e) => {
                col = col
                    .push(
                        text(format!("Error: {e}"))
                            .size(scaled(13.0, fs))
                            .color(theme.palette().danger),
                    )
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

fn file_row<'a>(
    fs: f32,
    label: &str,
    path: Option<&Path>,
    on_browse: Message,
    enabled: bool,
    theme: &Theme,
) -> Element<'a, Message> {
    let muted = muted_color(theme);

    let display_text: Element<'a, Message> = if let Some(name) = path.and_then(|p| p.file_name()) {
        text(name.to_string_lossy().to_string())
            .size(scaled(14.0, fs))
            .into()
    } else {
        text("No file selected")
            .size(scaled(14.0, fs))
            .color(muted)
            .into()
    };

    let btn = button(text("Browse").size(scaled(12.0, fs))).padding([6, 16]);
    let btn = if enabled {
        btn.on_press(on_browse).style(button::secondary)
    } else {
        btn.style(button::secondary)
    };

    let label_text = text(label.to_uppercase())
        .size(scaled(10.0, fs))
        .color(muted);

    let content = row![column![label_text, display_text,].width(Length::Fill), btn,]
        .spacing(8)
        .align_y(iced::Alignment::Center);

    container(content)
        .padding([12, 16])
        .style(container::rounded_box)
        .width(Length::Fill)
        .into()
}
