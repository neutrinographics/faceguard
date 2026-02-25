use std::path::Path;

use iced::widget::{button, column, container, progress_bar, row, text, Space};
use iced::{Element, Length, Theme};

use crate::app::{scaled, Message, ProcessingState};
use crate::widgets::drop_zone;
use crate::widgets::primary_button;
use crate::theme::{muted_color, tertiary_color};
use crate::widgets::faces_well::{self, FacesWellState};

pub fn view<'a>(
    fs: f32,
    input_path: Option<&Path>,
    output_path: Option<&Path>,
    processing: &ProcessingState,
    faces_well: &FacesWellState,
    theme: &Theme,
    browse_hovered: bool,
    drop_zone_hovered: bool,
    blur_button_hovered: bool,
) -> Element<'a, Message> {
    let muted = muted_color(theme);
    let tertiary = tertiary_color(theme);

    if input_path.is_none() {
        return drop_zone::view(fs, tertiary, theme, browse_hovered, drop_zone_hovered);
    }

    if let ProcessingState::Complete = processing {
        return complete_state(fs, muted, tertiary, output_path);
    }

    if let ProcessingState::Error(ref e) = processing {
        return error_state(fs, muted, tertiary, e);
    }

    workflow_view(fs, input_path, output_path, processing, faces_well, theme, blur_button_hovered)
}

fn complete_state<'a>(
    fs: f32,
    _muted: iced::Color,
    tertiary: iced::Color,
    output_path: Option<&Path>,
) -> Element<'a, Message> {
    let filename = output_path
        .and_then(|p| p.file_name())
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();

    centered(
        column![
            text("All done!").size(scaled(20.0, fs)),
            Space::new().height(6),
            text(format!("Saved as {filename}"))
                .size(scaled(14.0, fs))
                .color(tertiary),
            Space::new().height(28),
            button(text("Show in Folder").size(scaled(15.0, fs)))
                .on_press(Message::ShowInFolder)
                .padding([14, 24])
                .width(Length::Fill),
            Space::new().height(10),
            button(text("Blur Another File").size(scaled(14.0, fs)))
                .on_press(Message::StartOver)
                .padding([14, 20])
                .width(Length::Fill)
                .style(button::secondary),
        ]
        .align_x(iced::Alignment::Center)
        .width(280)
        .into(),
    )
}

fn error_state<'a>(
    fs: f32,
    _muted: iced::Color,
    tertiary: iced::Color,
    error: &str,
) -> Element<'a, Message> {
    centered(
        column![
            text("Something went wrong").size(scaled(18.0, fs)),
            Space::new().height(8),
            text(error.to_owned())
                .size(scaled(14.0, fs))
                .color(tertiary),
            Space::new().height(20),
            button(text("Try Again").size(scaled(14.0, fs)))
                .on_press(Message::RunBlur)
                .padding([14, 24])
                .width(Length::Fill),
            Space::new().height(10),
            button(text("Start Over").size(scaled(14.0, fs)))
                .on_press(Message::StartOver)
                .padding([14, 20])
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
    input_path: Option<&Path>,
    output_path: Option<&Path>,
    processing: &ProcessingState,
    faces_well: &FacesWellState,
    theme: &Theme,
    blur_button_hovered: bool,
) -> Element<'a, Message> {
    let muted = muted_color(theme);
    let tertiary = tertiary_color(theme);
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
            .push(Space::new().height(12))
            .push(file_row(
                fs,
                "Saves to",
                output_path,
                Message::SelectOutput,
                theme,
            ))
            .push(Space::new().height(20));
    }

    match processing {
        ProcessingState::Idle => {
            let blur_btn = primary_button::primary_button_fill(
                move || {
                    text("Blur All Faces")
                        .size(scaled(15.0, fs))
                        .color(iced::Color::WHITE)
                        .font(iced::Font {
                            weight: iced::font::Weight::Bold,
                            ..iced::Font::DEFAULT
                        })
                        .align_x(iced::Alignment::Center)
                        .width(Length::Fill)
                        .into()
                },
                Message::RunBlur,
                blur_button_hovered,
                Message::BlurButtonHover,
                [14, 24],
            );
            col = col
                .push(blur_btn)
                .push(Space::new().height(10))
                .push(
                    button(text("Choose Specific Faces\u{2026}").size(scaled(14.0, fs)))
                        .on_press(Message::RunPreview)
                        .padding([14, 20])
                        .width(Length::Fill)
                        .style(button::secondary),
                );
        }
        ProcessingState::Preparing => {
            col = col.push(progress_with_cancel(
                fs,
                muted,
                tertiary,
                "Preparing\u{2026}",
                None,
            ));
        }
        ProcessingState::Downloading(downloaded, total) => {
            let status = if *total > 0 {
                let pct = (*downloaded as f64 / *total as f64 * 100.0) as u32;
                format!("Downloading model \u{2014} {pct}%")
            } else {
                format!("Downloading model\u{2026} {} bytes", downloaded)
            };
            col = col.push(progress_with_cancel(fs, muted, tertiary, &status, None));
        }
        ProcessingState::Scanning(current, total) => {
            let (status, detail, pct) = frame_progress("Scanning", *current, *total);
            col = col.push(progress_with_cancel_detail(
                fs,
                muted,
                tertiary,
                &status,
                &detail,
                Some(pct),
            ));
        }
        ProcessingState::Previewed => {
            let blur_btn = primary_button::primary_button_fill(
                move || {
                    text("Blur Selected Faces")
                        .size(scaled(15.0, fs))
                        .color(iced::Color::WHITE)
                        .font(iced::Font {
                            weight: iced::font::Weight::Bold,
                            ..iced::Font::DEFAULT
                        })
                        .align_x(iced::Alignment::Center)
                        .width(Length::Fill)
                        .into()
                },
                Message::RunBlur,
                blur_button_hovered,
                Message::BlurButtonHover,
                [14, 24],
            );
            col = col
                .push(faces_well::view(faces_well, fs, theme))
                .push(Space::new().height(16))
                .push(
                    row![
                        blur_btn,
                        button(text("Re-scan").size(scaled(14.0, fs)))
                            .on_press(Message::RunPreview)
                            .padding([14, 20])
                            .style(button::secondary),
                    ]
                    .spacing(10),
                );
        }
        ProcessingState::Blurring(current, total) => {
            let (status, detail, pct) = frame_progress("Blurring faces", *current, *total);
            col = col.push(progress_with_cancel_detail(
                fs,
                muted,
                tertiary,
                &status,
                &detail,
                Some(pct),
            ));
        }
        _ => {}
    }

    col.into()
}

fn progress_with_cancel<'a>(
    fs: f32,
    _muted: iced::Color,
    tertiary: iced::Color,
    status: &str,
    progress: Option<f32>,
) -> Element<'a, Message> {
    let mut col = column![text(status.to_owned())
        .size(scaled(15.0, fs))
        .color(tertiary)]
    .spacing(8)
    .align_x(iced::Alignment::Center)
    .width(Length::Fill);

    if let Some(pct) = progress {
        col = col.push(progress_bar(0.0..=100.0, pct));
    }

    col = col.push(Space::new().height(16));
    col = col.push(
        button(text("Cancel").size(scaled(13.0, fs)))
            .on_press(Message::CancelWork)
            .padding([8, 20])
            .style(button::secondary),
    );

    container(col)
        .width(Length::Fill)
        .center_x(Length::Fill)
        .padding([48, 40])
        .into()
}

fn progress_with_cancel_detail<'a>(
    fs: f32,
    _muted: iced::Color,
    tertiary: iced::Color,
    status: &str,
    detail: &str,
    progress: Option<f32>,
) -> Element<'a, Message> {
    let mut col = column![text(status.to_owned()).size(scaled(15.0, fs))]
        .spacing(8)
        .align_x(iced::Alignment::Center)
        .width(Length::Fill);

    if let Some(pct) = progress {
        col = col.push(progress_bar(0.0..=100.0, pct));
    }

    col = col.push(
        text(detail.to_owned())
            .size(scaled(13.0, fs))
            .color(tertiary),
    );
    col = col.push(Space::new().height(16));
    col = col.push(
        button(text("Cancel").size(scaled(13.0, fs)))
            .on_press(Message::CancelWork)
            .padding([8, 20])
            .style(button::secondary),
    );

    container(col)
        .width(Length::Fill)
        .center_x(Length::Fill)
        .padding([48, 40])
        .into()
}

fn frame_progress(verb: &str, current: usize, total: usize) -> (String, String, f32) {
    let pct = if total > 0 {
        current as f32 / total as f32 * 100.0
    } else {
        0.0
    };
    let status = format!("{verb} \u{2014} {:.0}%", pct);
    let detail = if total > 0 {
        format!("Processing frame {current} of {total}\u{2026}")
    } else {
        format!("Processing frame {current}\u{2026}")
    };
    (status, detail, pct)
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
    let tertiary = tertiary_color(theme);

    let display_text: Element<'a, Message> = if let Some(name) = path.and_then(|p| p.file_name()) {
        text(name.to_string_lossy().to_string())
            .size(scaled(15.0, fs))
            .into()
    } else {
        text("No file selected")
            .size(scaled(15.0, fs))
            .color(tertiary)
            .into()
    };

    let btn = button(text("Change").size(scaled(13.0, fs)))
        .padding([6, 14])
        .on_press(on_browse)
        .style(button::secondary);

    let label_text = text(label.to_uppercase())
        .size(scaled(11.0, fs))
        .color(tertiary);

    let content = row![column![label_text, display_text].width(Length::Fill), btn]
        .spacing(8)
        .align_y(iced::Alignment::Center);

    container(content)
        .padding([14, 16])
        .style(container::rounded_box)
        .width(Length::Fill)
        .into()
}
