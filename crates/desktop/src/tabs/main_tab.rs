use std::path::Path;

use iced::widget::{button, column, container, progress_bar, row, svg, text, Space};
use iced::{Element, Length, Theme};

use crate::app::{scaled, Message, ProcessingState};
use crate::theme::{muted_color, tertiary_color};
use crate::widgets::drop_zone;
use crate::widgets::faces_well::{self, FacesWellState};
use crate::widgets::file_row;
use crate::widgets::primary_button;
use crate::widgets::secondary_button;

#[allow(clippy::too_many_arguments)]
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
    change_input_hovered: bool,
    change_output_hovered: bool,
    choose_faces_hovered: bool,
    cancel_hovered: bool,
    rescan_hovered: bool,
    face_card_hovered: &std::collections::HashSet<u32>,
    show_folder_hovered: bool,
    blur_another_hovered: bool,
) -> Element<'a, Message> {
    let muted = muted_color(theme);
    let tertiary = tertiary_color(theme);

    if input_path.is_none() {
        return drop_zone::view(fs, tertiary, theme, browse_hovered, drop_zone_hovered);
    }

    if let ProcessingState::Complete = processing {
        return complete_state(
            fs,
            muted,
            tertiary,
            output_path,
            theme,
            show_folder_hovered,
            blur_another_hovered,
        );
    }

    if let ProcessingState::Error(ref e) = processing {
        return error_state(fs, muted, tertiary, e);
    }

    workflow_view(
        fs,
        input_path,
        output_path,
        processing,
        faces_well,
        theme,
        blur_button_hovered,
        change_input_hovered,
        change_output_hovered,
        choose_faces_hovered,
        cancel_hovered,
        rescan_hovered,
        face_card_hovered,
    )
}

fn complete_state<'a>(
    fs: f32,
    _muted: iced::Color,
    tertiary: iced::Color,
    output_path: Option<&Path>,
    theme: &Theme,
    show_folder_hovered: bool,
    blur_another_hovered: bool,
) -> Element<'a, Message> {
    let filename = output_path
        .and_then(|p| p.file_name())
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();

    let success_bg = if is_dark_theme(theme) {
        iced::Color::from_rgba(
            0x2E as f32 / 255.0,
            0x8B as f32 / 255.0,
            0x57 as f32 / 255.0,
            0.15,
        )
    } else {
        iced::Color::from_rgb(
            0xE8 as f32 / 255.0,
            0xF5 as f32 / 255.0,
            0xEE as f32 / 255.0,
        )
    };

    let check_svg = svg(svg::Handle::from_memory(
        include_bytes!("../../assets/check.svg").as_slice(),
    ))
    .width(28)
    .height(28);

    let check_icon = container(check_svg)
        .width(64)
        .height(64)
        .center_x(64)
        .center_y(64)
        .style(move |_theme: &Theme| container::Style {
            background: Some(success_bg.into()),
            border: iced::border::Border {
                radius: 32.0.into(),
                ..iced::border::Border::default()
            },
            ..container::Style::default()
        });

    let show_btn = primary_button::primary_button_fill(
        move || {
            let folder_icon = svg(svg::Handle::from_memory(
                include_bytes!("../../assets/folder.svg").as_slice(),
            ))
            .width(16)
            .height(16);

            container(
                row![
                    folder_icon,
                    text("Show in Folder")
                        .size(scaled(16.0, fs))
                        .font(iced::Font {
                            weight: iced::font::Weight::Semibold,
                            ..iced::Font::DEFAULT
                        }),
                ]
                .spacing(8)
                .align_y(iced::Alignment::Center),
            )
            .width(Length::Fill)
            .center_x(Length::Fill)
            .into()
        },
        Message::ShowInFolder,
        show_folder_hovered,
        Message::ShowFolderHover,
        [14, 24],
    );

    let another_btn = secondary_button::secondary_button_fill(
        move || {
            text("Blur Another File")
                .size(scaled(16.0, fs))
                .font(iced::Font {
                    weight: iced::font::Weight::Semibold,
                    ..iced::Font::DEFAULT
                })
                .width(Length::Fill)
                .center()
                .into()
        },
        Message::StartOver,
        blur_another_hovered,
        Message::BlurAnotherHover,
        [14, 24],
    );

    centered(
        column![
            check_icon,
            Space::new().height(20),
            text("All done!")
                .size(scaled(22.0, fs))
                .font(iced::Font {
                    weight: iced::font::Weight::Semibold,
                    ..iced::Font::DEFAULT
                })
                .center(),
            Space::new().height(6),
            text(format!("Saved as {filename}"))
                .size(scaled(15.0, fs))
                .color(tertiary)
                .center(),
            Space::new().height(28),
            show_btn,
            Space::new().height(10),
            another_btn,
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
            text("Something went wrong").size(scaled(19.0, fs)),
            Space::new().height(8),
            text(error.to_owned())
                .size(scaled(15.0, fs))
                .color(tertiary),
            Space::new().height(20),
            button(text("Try Again").size(scaled(15.0, fs)))
                .on_press(Message::RunBlur)
                .padding([14, 24])
                .width(Length::Fill),
            Space::new().height(10),
            button(text("Start Over").size(scaled(15.0, fs)))
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

#[allow(clippy::too_many_arguments)]
fn workflow_view<'a>(
    fs: f32,
    input_path: Option<&Path>,
    output_path: Option<&Path>,
    processing: &ProcessingState,
    faces_well: &FacesWellState,
    theme: &Theme,
    blur_button_hovered: bool,
    change_input_hovered: bool,
    change_output_hovered: bool,
    choose_faces_hovered: bool,
    cancel_hovered: bool,
    rescan_hovered: bool,
    face_card_hovered: &std::collections::HashSet<u32>,
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
            .push(file_row::file_row(
                fs,
                "Input",
                input_path,
                Message::SelectInput,
                change_input_hovered,
                Message::ChangeInputHover,
                theme,
            ))
            .push(Space::new().height(12))
            .push(file_row::file_row(
                fs,
                "Saves to",
                output_path,
                Message::SelectOutput,
                change_output_hovered,
                Message::ChangeOutputHover,
                theme,
            ))
            .push(Space::new().height(20));
    }

    match processing {
        ProcessingState::Idle => {
            let blur_btn = primary_button::primary_button_fill(
                move || {
                    let blur_icon = svg(svg::Handle::from_memory(
                        include_bytes!("../../assets/blur.svg").as_slice(),
                    ))
                    .width(16)
                    .height(16)
                    .style(|_theme: &Theme, _status| svg::Style {
                        color: Some(iced::Color::WHITE),
                    });

                    container(
                        row![
                            blur_icon,
                            text("Blur All Faces")
                                .size(scaled(16.0, fs))
                                .color(iced::Color::WHITE)
                                .font(iced::Font {
                                    weight: iced::font::Weight::Bold,
                                    ..iced::Font::DEFAULT
                                }),
                        ]
                        .spacing(8)
                        .align_y(iced::Alignment::Center),
                    )
                    .width(Length::Fill)
                    .center_x(Length::Fill)
                    .into()
                },
                Message::RunBlur,
                blur_button_hovered,
                Message::BlurButtonHover,
                [14, 24],
            );
            col = col.push(blur_btn).push(Space::new().height(10)).push(
                secondary_button::secondary_button_fill(
                    move || {
                        text("Choose Specific Faces\u{2026}")
                            .size(scaled(15.0, fs))
                            .font(iced::Font {
                                weight: iced::font::Weight::Bold,
                                ..iced::Font::DEFAULT
                            })
                            .align_x(iced::Alignment::Center)
                            .width(Length::Fill)
                            .into()
                    },
                    Message::RunPreview,
                    choose_faces_hovered,
                    Message::ChooseFacesHover,
                    [14, 20],
                ),
            );
        }
        ProcessingState::Preparing => {
            col = col.push(progress_with_cancel(
                fs,
                muted,
                tertiary,
                "Preparing\u{2026}",
                None,
                cancel_hovered,
            ));
        }
        ProcessingState::Downloading(downloaded, total) => {
            let status = if *total > 0 {
                let pct = (*downloaded as f64 / *total as f64 * 100.0) as u32;
                format!("Downloading face detection model \u{2014} {pct}%")
            } else {
                format!(
                    "Downloading face detection model\u{2026} {} bytes",
                    downloaded
                )
            };
            col = col.push(progress_with_cancel(
                fs,
                muted,
                tertiary,
                &status,
                None,
                cancel_hovered,
            ));
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
                cancel_hovered,
            ));
        }
        ProcessingState::Previewed => {
            let blur_btn = primary_button::primary_button_fill(
                move || {
                    let blur_icon = svg(svg::Handle::from_memory(
                        include_bytes!("../../assets/blur.svg").as_slice(),
                    ))
                    .width(16)
                    .height(16)
                    .style(|_theme: &Theme, _status| svg::Style {
                        color: Some(iced::Color::WHITE),
                    });

                    container(
                        row![
                            blur_icon,
                            text("Blur Selected Faces")
                                .size(scaled(16.0, fs))
                                .color(iced::Color::WHITE)
                                .font(iced::Font {
                                    weight: iced::font::Weight::Bold,
                                    ..iced::Font::DEFAULT
                                }),
                        ]
                        .spacing(8)
                        .align_y(iced::Alignment::Center),
                    )
                    .width(Length::Fill)
                    .center_x(Length::Fill)
                    .into()
                },
                Message::RunBlur,
                blur_button_hovered,
                Message::BlurButtonHover,
                [14, 24],
            );
            col = col
                .push(faces_well::view(faces_well, fs, theme, face_card_hovered))
                .push(Space::new().height(16))
                .push(
                    row![
                        blur_btn,
                        secondary_button::secondary_button(
                            move || text("Re-scan").size(scaled(15.0, fs)).into(),
                            Message::RunPreview,
                            rescan_hovered,
                            Message::RescanHover,
                            [14, 20],
                        ),
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
                cancel_hovered,
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
    cancel_hovered: bool,
) -> Element<'a, Message> {
    let mut col = column![text(status.to_owned())
        .size(scaled(16.0, fs))
        .color(tertiary)]
    .spacing(8)
    .align_x(iced::Alignment::Center)
    .width(Length::Fill);

    if let Some(pct) = progress {
        col = col.push(styled_progress_bar(pct));
    }

    col = col.push(Space::new().height(16));
    col = col.push(secondary_button::secondary_button_small(
        move || text("Cancel").size(scaled(14.0, fs)).into(),
        Message::CancelWork,
        cancel_hovered,
        Message::CancelHover,
        [8, 20],
    ));

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
    cancel_hovered: bool,
) -> Element<'a, Message> {
    let mut col = column![text(status.to_owned()).size(scaled(16.0, fs))]
        .spacing(8)
        .align_x(iced::Alignment::Center)
        .width(Length::Fill);

    if let Some(pct) = progress {
        col = col.push(styled_progress_bar(pct));
    }

    col = col.push(
        text(detail.to_owned())
            .size(scaled(14.0, fs))
            .color(tertiary),
    );
    col = col.push(Space::new().height(16));
    col = col.push(secondary_button::secondary_button_small(
        move || text("Cancel").size(scaled(14.0, fs)).into(),
        Message::CancelWork,
        cancel_hovered,
        Message::CancelHover,
        [8, 20],
    ));

    container(col)
        .width(Length::Fill)
        .center_x(Length::Fill)
        .padding([48, 40])
        .into()
}

fn styled_progress_bar(pct: f32) -> Element<'static, Message> {
    progress_bar(0.0..=100.0, pct)
        .girth(8.0)
        .style(|theme: &Theme| {
            let palette = theme.palette();
            let luma = palette.background.r * 0.299
                + palette.background.g * 0.587
                + palette.background.b * 0.114;
            let track_bg = if luma > 0.5 {
                iced::Color::from_rgb(
                    0xF0 as f32 / 255.0,
                    0xED as f32 / 255.0,
                    0xE8 as f32 / 255.0,
                )
            } else {
                iced::Color {
                    r: (palette.background.r + 0.12).min(1.0),
                    g: (palette.background.g + 0.12).min(1.0),
                    b: (palette.background.b + 0.12).min(1.0),
                    a: 1.0,
                }
            };
            progress_bar::Style {
                background: track_bg.into(),
                bar: palette.primary.into(),
                border: iced::border::Border {
                    radius: 100.0.into(),
                    ..iced::border::Border::default()
                },
            }
        })
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

fn is_dark_theme(theme: &Theme) -> bool {
    let p = theme.palette();
    let luma = p.background.r * 0.299 + p.background.g * 0.587 + p.background.b * 0.114;
    luma <= 0.5
}
