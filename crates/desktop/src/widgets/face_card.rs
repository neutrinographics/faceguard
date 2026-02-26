use std::path::{Path, PathBuf};
use std::time::Duration;

use iced::widget::{button, container, image, mouse_area, row, stack, text, Space};
use iced::{Color, Element, Length, Theme};
use iced_anim::transition::Easing;
use iced_anim::AnimationBuilder;

use crate::app::{scaled, Message};

const CORNER_RADIUS: f32 = 12.0;
const BORDER_WIDTH: f32 = 2.5;
const BADGE_RADIUS: f32 = 10.0;
const CHECK_SIZE: f32 = 20.0;
const SCALE_GROW: f32 = 2.0;
const ANIMATION_DURATION: Duration = Duration::from_millis(200);

/// The outer size each card occupies in the grid (fixed, never changes with hover).
pub const FULL_CARD_SIZE: f32 = 109.0;

/// Image size at rest. Leaves room for border + padding on each side.
const IMG_SIZE: f32 = FULL_CARD_SIZE - BORDER_WIDTH * 4.0;

pub fn face_card<'a>(
    path: &Path,
    selected: bool,
    on_press: Message,
    badge: Option<String>,
    hovered: bool,
    card_id: u32,
    fs: f32,
    theme: &Theme,
) -> Element<'a, Message> {
    let palette = theme.palette();
    let surface_alt = surface_alt_color(theme);
    let path_buf = path.to_path_buf();

    let target = if hovered { 1.0_f32 } else { 0.0 };

    let animated: Element<'a, Message> = AnimationBuilder::new(target, move |t: f32| {
        let t = t.clamp(0.0, 1.0);
        build_card(
            &path_buf,
            selected,
            &on_press,
            &badge,
            t,
            palette,
            surface_alt,
            fs,
        )
    })
    .animates_layout(true)
    .animation(Easing::EASE_OUT.with_duration(ANIMATION_DURATION))
    .into();

    // Fixed-size outer container establishes grid footprint.
    // The animated card inside can overflow visually when hovered.
    let outer = container(
        mouse_area(animated)
            .on_enter(Message::FaceCardHover(card_id, true))
            .on_exit(Message::FaceCardHover(card_id, false)),
    )
    .width(FULL_CARD_SIZE)
    .height(FULL_CARD_SIZE)
    .center_x(FULL_CARD_SIZE)
    .center_y(FULL_CARD_SIZE);

    outer.into()
}

fn build_card<'a>(
    path: &PathBuf,
    selected: bool,
    on_press: &Message,
    badge: &Option<String>,
    hover_amount: f32,
    palette: iced::theme::Palette,
    surface_alt: Color,
    fs: f32,
) -> Element<'a, Message> {
    let grow = SCALE_GROW * hover_amount;
    let card_size = IMG_SIZE + grow * 2.0;

    let inner_radius = (CORNER_RADIUS - BORDER_WIDTH).max(0.0);
    let img = image(image::Handle::from_path(path))
        .width(card_size)
        .height(card_size)
        .border_radius(inner_radius);

    // Build overlay elements
    let mut items: Vec<Element<'a, Message>> = Vec::new();

    if selected {
        let check: Element<'a, Message> = container(
            text("\u{2713}")
                .size(scaled(11.0, fs))
                .color(Color::WHITE)
                .align_x(iced::Alignment::Center),
        )
        .width(CHECK_SIZE)
        .height(CHECK_SIZE)
        .center_x(CHECK_SIZE)
        .center_y(CHECK_SIZE)
        .style(move |_theme: &Theme| container::Style {
            background: Some(palette.primary.into()),
            border: iced::border::Border {
                radius: BADGE_RADIUS.into(),
                ..iced::border::Border::default()
            },
            ..container::Style::default()
        })
        .into();
        items.push(check);
    }

    items.push(Space::new().width(Length::Fill).into());

    if let Some(badge_text) = badge {
        let count: Element<'a, Message> = container(
            text(badge_text.clone())
                .size(scaled(11.0, fs))
                .color(Color::WHITE)
                .font(iced::Font {
                    weight: iced::font::Weight::Semibold,
                    ..iced::Font::DEFAULT
                }),
        )
        .padding([2, 6])
        .style(|_theme: &Theme| container::Style {
            background: Some(Color::from_rgba(0.0, 0.0, 0.0, 0.55).into()),
            border: iced::border::Border {
                radius: BADGE_RADIUS.into(),
                ..iced::border::Border::default()
            },
            ..container::Style::default()
        })
        .into();
        items.push(count);
    }

    let overlay_row = row(items).width(Length::Fill);

    let overlay = container(overlay_row)
        .width(Length::Fill)
        .height(Length::Fill)
        .padding(6)
        .align_y(iced::alignment::Vertical::Bottom);

    let overlay_el: Element<'a, Message> = overlay.into();
    let card_stack = stack![img, overlay_el];

    let border_color = if selected {
        palette.primary
    } else {
        Color {
            a: 0.15,
            ..palette.text
        }
    };

    // Styled container provides the visible border, radius, and background.
    // The button inside is fully transparent — just for click handling.
    let btn = button(card_stack)
        .on_press(on_press.clone())
        .padding(0)
        .style(|_theme: &Theme, _status: button::Status| button::Style {
            background: None,
            border: iced::border::Border::default(),
            ..button::Style::default()
        });

    container(btn)
        .padding(iced::Padding::from(BORDER_WIDTH))
        .style(move |_theme: &Theme| container::Style {
            background: Some(surface_alt.into()),
            border: iced::border::Border {
                color: border_color,
                width: BORDER_WIDTH,
                radius: CORNER_RADIUS.into(),
            },
            ..container::Style::default()
        })
        .into()
}

fn surface_alt_color(theme: &Theme) -> Color {
    let p = theme.palette();
    let luma = p.background.r * 0.299 + p.background.g * 0.587 + p.background.b * 0.114;
    if luma > 0.5 {
        Color::from_rgb(
            0xF0 as f32 / 255.0,
            0xED as f32 / 255.0,
            0xE8 as f32 / 255.0,
        )
    } else {
        Color {
            r: (p.background.r + 0.12).min(1.0),
            g: (p.background.g + 0.12).min(1.0),
            b: (p.background.b + 0.12).min(1.0),
            a: 1.0,
        }
    }
}
