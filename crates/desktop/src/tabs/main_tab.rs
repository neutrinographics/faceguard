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

    if input_path.is_none() {
        return empty_state(fs, muted);
    }

    if let ProcessingState::Complete = processing {
        return complete_state(fs, muted, output_path);
    }

    if let ProcessingState::Error(ref e) = processing {
        return error_state(fs, muted, e);
    }

    workflow_view(
        fs,
        muted,
        input_path,
        output_path,
        processing,
        faces_well,
        theme,
    )
}

fn empty_state(fs: f32, muted: iced::Color) -> Element<'static, Message> {
    centered(
        column![
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
        .align_x(iced::Alignment::Center)
        .into(),
    )
}

fn complete_state<'a>(
    fs: f32,
    muted: iced::Color,
    output_path: Option<&Path>,
) -> Element<'a, Message> {
    let filename = output_path
        .and_then(|p| p.file_name())
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();

    centered(
        column![
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
        .width(280)
        .into(),
    )
}

fn error_state<'a>(fs: f32, muted: iced::Color, error: &str) -> Element<'a, Message> {
    centered(
        column![
            text("Something went wrong").size(scaled(18.0, fs)),
            Space::new().height(8),
            text(error.to_owned()).size(scaled(14.0, fs)).color(muted),
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
        .width(280)
        .into(),
    )
}

fn workflow_view<'a>(
    fs: f32,
    muted: iced::Color,
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

    let mut col = column![].spacing(0);

    if !is_processing {
        col = col
            .push(file_row(
                fs,
                "Input",
                input_path,
                Message::SelectInput,
                theme,
            ))
            .push(Space::new().height(6))
            .push(file_row(
                fs,
                "Output",
                output_path,
                Message::SelectOutput,
                theme,
            ))
            .push(Space::new().height(16));
    }

    match processing {
        ProcessingState::Idle => {
            col = col
                .push(
                    button(text("Blur All Faces").size(scaled(14.0, fs)))
                        .on_press(Message::RunBlur)
                        .padding([12, 32])
                        .width(Length::Fill),
                )
                .push(Space::new().height(8))
                .push(
                    button(text("Choose Specific Faces...").size(scaled(14.0, fs)))
                        .on_press(Message::RunPreview)
                        .padding([12, 32])
                        .width(Length::Fill)
                        .style(button::secondary),
                );
        }
        ProcessingState::Preparing => {
            col = col.push(progress_with_cancel(fs, muted, "Preparing...", None));
        }
        ProcessingState::Downloading(downloaded, total) => {
            let status = if *total > 0 {
                let pct = (*downloaded as f64 / *total as f64 * 100.0) as u32;
                format!("Downloading model... {pct}%")
            } else {
                format!("Downloading model... {} bytes", downloaded)
            };
            col = col.push(progress_with_cancel(fs, muted, &status, None));
        }
        ProcessingState::Scanning(current, total) => {
            let (status, pct) = frame_progress("Scanning", *current, *total);
            col = col.push(progress_with_cancel(fs, muted, &status, Some(pct)));
        }
        ProcessingState::Previewed => {
            col = col
                .push(faces_well::view(faces_well, fs, theme))
                .push(Space::new().height(12))
                .push(
                    button(text("Blur Selected Faces").size(scaled(14.0, fs)))
                        .on_press(Message::RunBlur)
                        .padding([12, 32])
                        .width(Length::Fill),
                )
                .push(Space::new().height(8))
                .push(
                    button(text("Re-scan").size(scaled(13.0, fs)))
                        .on_press(Message::RunPreview)
                        .style(button::text),
                );
        }
        ProcessingState::Blurring(current, total) => {
            let (status, pct) = frame_progress("Processing", *current, *total);
            col = col.push(progress_with_cancel(fs, muted, &status, Some(pct)));
        }
        _ => {}
    }

    col.into()
}

fn progress_with_cancel<'a>(
    fs: f32,
    muted: iced::Color,
    status: &str,
    progress: Option<f32>,
) -> Element<'a, Message> {
    let mut col = column![text(status.to_owned()).size(scaled(14.0, fs)).color(muted)].spacing(8);

    if let Some(pct) = progress {
        col = col.push(progress_bar(0.0..=100.0, pct));
    }

    col.push(
        button(text("Cancel").size(scaled(13.0, fs)))
            .on_press(Message::CancelWork)
            .padding([8, 24])
            .style(button::secondary),
    )
    .into()
}

fn frame_progress(verb: &str, current: usize, total: usize) -> (String, f32) {
    let pct = if total > 0 {
        current as f32 / total as f32 * 100.0
    } else {
        0.0
    };
    let status = if total > 0 {
        format!("{verb} frame {current}/{total}")
    } else {
        format!("{verb} frame {current}...")
    };
    (status, pct)
}

fn centered(content: Element<'_, Message>) -> Element<'_, Message> {
    container(content)
        .width(Length::Fill)
        .height(Length::Fill)
        .center_x(Length::Fill)
        .center_y(Length::Fill)
        .into()
}

fn file_row<'a>(
    fs: f32,
    label: &str,
    path: Option<&Path>,
    on_browse: Message,
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

    let btn = button(text("Change").size(scaled(12.0, fs)))
        .padding([6, 16])
        .on_press(on_browse)
        .style(button::secondary);

    let label_text = text(label.to_uppercase())
        .size(scaled(10.0, fs))
        .color(muted);

    let content = row![column![label_text, display_text].width(Length::Fill), btn]
        .spacing(8)
        .align_y(iced::Alignment::Center);

    container(content)
        .padding([12, 16])
        .style(container::rounded_box)
        .width(Length::Fill)
        .into()
}
