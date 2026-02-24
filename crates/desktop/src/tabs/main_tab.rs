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
    let muted = muted_color(theme);

    // No file selected — empty state
    if input_path.is_none() {
        let empty = column![
            text("Drop a file here or click Browse")
                .size(scaled(14.0, fs))
                .color(muted),
            Space::new().height(12),
            button(text("Browse Files").size(scaled(14.0, fs)))
                .on_press(Message::SelectInput)
                .padding([12, 32]),
            Space::new().height(8),
            text("MP4, MOV, AVI, JPG, PNG")
                .size(scaled(12.0, fs))
                .color(muted),
        ]
        .align_x(iced::Alignment::Center);

        return container(empty)
            .width(Length::Fill)
            .height(Length::Fill)
            .center_x(Length::Fill)
            .center_y(Length::Fill)
            .into();
    }

    // Complete state — centered
    if let ProcessingState::Complete = processing {
        let filename = output_path
            .and_then(|p| p.file_name())
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();

        let done = column![
            text("\u{2713} Complete").size(scaled(18.0, fs)),
            Space::new().height(4),
            text(filename).size(scaled(14.0, fs)).color(muted),
            Space::new().height(16),
            button(text("Show in Folder").size(scaled(14.0, fs)))
                .on_press(Message::ShowInFolder)
                .padding([12, 32])
                .width(Length::Fill),
            Space::new().height(8),
            button(text("Blur Another File").size(scaled(14.0, fs)))
                .on_press(Message::StartOver)
                .padding([12, 32])
                .width(Length::Fill)
                .style(button::secondary),
        ]
        .align_x(iced::Alignment::Center)
        .width(280);

        return container(done)
            .width(Length::Fill)
            .height(Length::Fill)
            .center_x(Length::Fill)
            .center_y(Length::Fill)
            .into();
    }

    // Error state — centered
    if let ProcessingState::Error(ref e) = processing {
        let err = column![
            text("Something went wrong").size(scaled(18.0, fs)),
            Space::new().height(8),
            text(e.clone()).size(scaled(14.0, fs)).color(muted),
            Space::new().height(16),
            button(text("Try Again").size(scaled(14.0, fs)))
                .on_press(Message::RunBlur)
                .padding([12, 32])
                .width(Length::Fill),
            Space::new().height(8),
            button(text("Start Over").size(scaled(14.0, fs)))
                .on_press(Message::StartOver)
                .padding([12, 32])
                .width(Length::Fill)
                .style(button::secondary),
        ]
        .align_x(iced::Alignment::Center)
        .width(280);

        return container(err)
            .width(Length::Fill)
            .height(Length::Fill)
            .center_x(Length::Fill)
            .center_y(Length::Fill)
            .into();
    }

    let is_processing = matches!(
        processing,
        ProcessingState::Preparing
            | ProcessingState::Downloading(..)
            | ProcessingState::Scanning(..)
            | ProcessingState::Blurring(..)
    );

    let mut col = column![].spacing(0);

    // File rows (hidden during processing for cleaner focus)
    if !is_processing {
        col = col
            .push(file_row(
                fs,
                "Input",
                input_path,
                Message::SelectInput,
                true,
                theme,
            ))
            .push(Space::new().height(6))
            .push(file_row(
                fs,
                "Output",
                output_path,
                Message::SelectOutput,
                true,
                theme,
            ))
            .push(Space::new().height(16));
    }

    match processing {
        ProcessingState::Idle => {
            col = col.push(
                button(text("Blur All Faces").size(scaled(14.0, fs)))
                    .on_press(Message::RunBlur)
                    .padding([12, 32])
                    .width(Length::Fill),
            );
            col = col.push(Space::new().height(8));
            col = col.push(
                button(
                    text("Choose Specific Faces...").size(scaled(14.0, fs)),
                )
                .on_press(Message::RunPreview)
                .padding([12, 32])
                .width(Length::Fill)
                .style(button::secondary),
            );
        }
        ProcessingState::Preparing => {
            col = col
                .push(
                    text("Loading model...")
                        .size(scaled(14.0, fs))
                        .color(muted),
                )
                .push(Space::new().height(8))
                .push(
                    button(text("Cancel").size(scaled(13.0, fs)))
                        .on_press(Message::CancelWork)
                        .padding([8, 24])
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
                .push(text(status).size(scaled(14.0, fs)).color(muted))
                .push(Space::new().height(8))
                .push(
                    button(text("Cancel").size(scaled(13.0, fs)))
                        .on_press(Message::CancelWork)
                        .padding([8, 24])
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
                .push(text(status).size(scaled(14.0, fs)).color(muted))
                .push(Space::new().height(8))
                .push(progress_bar(0.0..=100.0, pct))
                .push(Space::new().height(8))
                .push(
                    button(text("Cancel").size(scaled(13.0, fs)))
                        .on_press(Message::CancelWork)
                        .padding([8, 24])
                        .style(button::secondary),
                );
        }
        ProcessingState::Previewed => {
            col = col.push(faces_well::view(faces_well, fs, theme));
            col = col.push(Space::new().height(12));
            col = col.push(
                button(text("Blur Selected Faces").size(scaled(14.0, fs)))
                    .on_press(Message::RunBlur)
                    .padding([12, 32])
                    .width(Length::Fill),
            );
            col = col.push(Space::new().height(8));
            col = col.push(
                button(text("Re-scan").size(scaled(13.0, fs)))
                    .on_press(Message::RunPreview)
                    .style(button::text),
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
                .push(text(status).size(scaled(14.0, fs)).color(muted))
                .push(Space::new().height(8))
                .push(progress_bar(0.0..=100.0, pct))
                .push(Space::new().height(8))
                .push(
                    button(text("Cancel").size(scaled(13.0, fs)))
                        .on_press(Message::CancelWork)
                        .padding([8, 24])
                        .style(button::secondary),
                );
        }
        // Complete and Error handled above as early returns
        _ => {}
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

    let btn = button(text("Change").size(scaled(12.0, fs))).padding([6, 16]);
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
