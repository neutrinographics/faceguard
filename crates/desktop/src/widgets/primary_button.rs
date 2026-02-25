use std::time::Duration;

use iced::border::Border;
use iced::widget::{button, container, mouse_area};
use iced::{Color, Element, Padding, Theme};
use iced_anim::transition::Easing;
use iced_anim::AnimationBuilder;

/// Creates an animated primary button that floats up slightly and darkens on hover.
///
/// - `content`: closure that builds the button's inner content (called on each frame)
/// - `on_press`: message sent when the button is clicked
/// - `hovered`: whether the button is currently hovered (tracked externally)
/// - `on_hover`: message factory — called with `true` on mouse enter, `false` on exit
/// - `padding`: button padding as `[vertical, horizontal]`
pub fn primary_button<'a, Message: Clone + 'a>(
    content: impl Fn() -> Element<'a, Message> + 'a,
    on_press: Message,
    hovered: bool,
    on_hover: impl Fn(bool) -> Message + 'a,
    padding: [u16; 2],
) -> Element<'a, Message> {
    let target_offset: f32 = if hovered { 1.0 } else { 0.0 };

    let animated: Element<'a, Message> = AnimationBuilder::new(target_offset, move |offset: f32| {
        let t = offset.clamp(0.0, 1.0);

        let btn = button(content())
            .on_press(on_press.clone())
            .padding(padding)
            .style(move |theme: &Theme, status: button::Status| {
                let palette = theme.extended_palette();
                let base = palette.primary.base.color;

                let hover_amount = match status {
                    button::Status::Pressed => 1.0_f32,
                    _ => t,
                };

                let bg = Color {
                    r: (base.r - 0.05 * hover_amount).max(0.0),
                    g: (base.g - 0.05 * hover_amount).max(0.0),
                    b: (base.b - 0.05 * hover_amount).max(0.0),
                    a: 1.0,
                };

                button::Style {
                    background: Some(iced::Background::Color(bg)),
                    text_color: Color::WHITE,
                    border: Border {
                        radius: 8.0.into(),
                        ..Border::default()
                    },
                    ..button::Style::default()
                }
            });

        let top = (3.0 - offset * 3.0).max(0.0);
        let bottom = (offset * 3.0).min(3.0);
        container(btn)
            .padding(Padding {
                top,
                right: 0.0,
                bottom,
                left: 0.0,
            })
            .into()
    })
    .animates_layout(true)
    .animation(Easing::EASE_OUT.with_duration(Duration::from_millis(200)))
    .into();

    mouse_area(animated)
        .on_enter(on_hover(true))
        .on_exit(on_hover(false))
        .into()
}
