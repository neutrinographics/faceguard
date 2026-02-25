use std::time::Duration;

use iced::border::Border;
use iced::widget::{button, container, mouse_area};
use iced::{Color, Element, Padding, Theme};
use iced_anim::transition::Easing;
use iced_anim::AnimationBuilder;

const HOVER_DARKEN: f32 = 0.05;
const FLOAT_HEIGHT: f32 = 3.0;
const CORNER_RADIUS: f32 = 8.0;
const ANIMATION_DURATION: Duration = Duration::from_millis(200);

pub fn primary_button<'a, Message: Clone + 'a>(
    content: impl Fn() -> Element<'a, Message> + 'a,
    on_press: Message,
    hovered: bool,
    on_hover: impl Fn(bool) -> Message + 'a,
    padding: [u16; 2],
) -> Element<'a, Message> {
    let target = if hovered { 1.0_f32 } else { 0.0 };

    let animated: Element<'a, Message> = AnimationBuilder::new(target, move |t: f32| {
        let t = t.clamp(0.0, 1.0);
        build_button(&content, &on_press, padding, t)
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
    hover_amount: f32,
) -> Element<'a, Message> {
    let btn = button(content())
        .on_press(on_press.clone())
        .padding(padding)
        .style(move |theme: &Theme, status: button::Status| {
            let base = theme.extended_palette().primary.base.color;
            let amount = if status == button::Status::Pressed {
                1.0
            } else {
                hover_amount
            };
            styled(base, amount)
        });

    let rise = hover_amount * FLOAT_HEIGHT;
    container(btn)
        .padding(Padding {
            top: FLOAT_HEIGHT - rise,
            bottom: rise,
            ..Padding::ZERO
        })
        .into()
}

fn styled(base: Color, hover_amount: f32) -> button::Style {
    button::Style {
        background: Some(darken(base, hover_amount).into()),
        text_color: Color::WHITE,
        border: Border {
            radius: CORNER_RADIUS.into(),
            ..Border::default()
        },
        ..button::Style::default()
    }
}

fn darken(color: Color, amount: f32) -> Color {
    let shift = HOVER_DARKEN * amount;
    Color {
        r: (color.r - shift).max(0.0),
        g: (color.g - shift).max(0.0),
        b: (color.b - shift).max(0.0),
        a: 1.0,
    }
}
