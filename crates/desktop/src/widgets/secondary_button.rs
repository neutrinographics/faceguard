use std::time::Duration;

use iced::border::Border;
use iced::widget::{button, container, mouse_area};
use iced::{Color, Element, Length, Padding, Theme};
use iced_anim::transition::Easing;
use iced_anim::AnimationBuilder;

use crate::theme::{surface_color, tertiary_color};

const CORNER_RADIUS: f32 = 12.0;
const CORNER_RADIUS_SM: f32 = 8.0;
const ANIMATION_DURATION: Duration = Duration::from_millis(200);

pub fn secondary_button<'a, Message: Clone + 'a>(
    content: impl Fn() -> Element<'a, Message> + 'a,
    on_press: Message,
    hovered: bool,
    on_hover: impl Fn(bool) -> Message + 'a,
    padding: [u16; 2],
) -> Element<'a, Message> {
    secondary_button_inner(content, on_press, hovered, on_hover, padding, Length::Shrink, CORNER_RADIUS)
}

pub fn secondary_button_fill<'a, Message: Clone + 'a>(
    content: impl Fn() -> Element<'a, Message> + 'a,
    on_press: Message,
    hovered: bool,
    on_hover: impl Fn(bool) -> Message + 'a,
    padding: [u16; 2],
) -> Element<'a, Message> {
    secondary_button_inner(content, on_press, hovered, on_hover, padding, Length::Fill, CORNER_RADIUS)
}

pub fn secondary_button_small<'a, Message: Clone + 'a>(
    content: impl Fn() -> Element<'a, Message> + 'a,
    on_press: Message,
    hovered: bool,
    on_hover: impl Fn(bool) -> Message + 'a,
    padding: [u16; 2],
) -> Element<'a, Message> {
    secondary_button_inner(content, on_press, hovered, on_hover, padding, Length::Shrink, CORNER_RADIUS_SM)
}

fn secondary_button_inner<'a, Message: Clone + 'a>(
    content: impl Fn() -> Element<'a, Message> + 'a,
    on_press: Message,
    hovered: bool,
    on_hover: impl Fn(bool) -> Message + 'a,
    padding: [u16; 2],
    width: Length,
    radius: f32,
) -> Element<'a, Message> {
    let target = if hovered { 1.0_f32 } else { 0.0 };

    let animated: Element<'a, Message> = AnimationBuilder::new(target, move |t: f32| {
        let t = t.clamp(0.0, 1.0);
        build_button(&content, &on_press, padding, width, radius, t)
    })
    .animates_layout(true)
    .animation(Easing::EASE_OUT.with_duration(ANIMATION_DURATION))
    .into();

    mouse_area(animated)
        .on_enter(on_hover(true))
        .on_exit(on_hover(false))
        .into()
}

fn build_button<'a, Message: Clone + 'a>(
    content: &dyn Fn() -> Element<'a, Message>,
    on_press: &Message,
    padding: [u16; 2],
    width: Length,
    radius: f32,
    hover_amount: f32,
) -> Element<'a, Message> {
    let btn = button(content())
        .on_press(on_press.clone())
        .padding(padding)
        .width(width)
        .style(move |theme: &Theme, status: button::Status| {
            let amount = if status == button::Status::Pressed {
                1.0
            } else {
                hover_amount
            };
            styled(theme, radius, amount)
        });

    container(btn)
        .padding(Padding::ZERO)
        .into()
}

fn styled(theme: &Theme, radius: f32, hover_amount: f32) -> button::Style {
    let surface = surface_color(theme);
    let surface_alt = surface_alt_color(theme);
    let bg = lerp_color(surface, surface_alt, hover_amount);

    let border_base = border_color(theme);
    let border_hover = tertiary_color(theme);
    let border = lerp_color(border_base, border_hover, hover_amount);

    let text = muted_text_color(theme);

    button::Style {
        background: Some(bg.into()),
        text_color: text,
        border: Border {
            color: border,
            width: 1.0,
            radius: radius.into(),
        },
        ..button::Style::default()
    }
}

fn surface_alt_color(theme: &Theme) -> Color {
    let p = theme.palette();
    let luma = p.background.r * 0.299 + p.background.g * 0.587 + p.background.b * 0.114;
    if luma > 0.5 {
        // Light theme: a warm off-white
        Color::from_rgb(0xF0 as f32 / 255.0, 0xED as f32 / 255.0, 0xE8 as f32 / 255.0)
    } else {
        // Dark theme: slightly lighter than surface
        Color {
            r: (p.background.r + 0.12).min(1.0),
            g: (p.background.g + 0.12).min(1.0),
            b: (p.background.b + 0.12).min(1.0),
            a: 1.0,
        }
    }
}

fn border_color(theme: &Theme) -> Color {
    let p = theme.palette();
    Color { a: 0.15, ..p.text }
}

fn muted_text_color(theme: &Theme) -> Color {
    let p = theme.palette();
    Color { a: 0.55, ..p.text }
}

fn lerp_color(a: Color, b: Color, t: f32) -> Color {
    Color {
        r: a.r + (b.r - a.r) * t,
        g: a.g + (b.g - a.g) * t,
        b: a.b + (b.b - a.b) * t,
        a: a.a + (b.a - a.a) * t,
    }
}
