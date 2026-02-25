use iced::border::Border;
use iced::widget::button;
use iced::{Color, Element, Theme};

/// A button style function that renders a rounded primary-colored button
/// with white text, matching the mockup's `.browse-btn` / `.btn-primary` style.
pub fn primary_button_style(theme: &Theme, status: button::Status) -> button::Style {
    let palette = theme.extended_palette();
    let base = palette.primary.base.color;

    let bg = match status {
        button::Status::Hovered | button::Status::Pressed => Color {
            r: (base.r - 0.05).max(0.0),
            g: (base.g - 0.05).max(0.0),
            b: (base.b - 0.05).max(0.0),
            a: 1.0,
        },
        button::Status::Disabled => Color {
            a: 0.5,
            ..base
        },
        _ => base,
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
}

/// Convenience: creates a styled primary button from content.
pub fn primary_button<'a, Message: 'a>(
    content: impl Into<Element<'a, Message>>,
) -> button::Button<'a, Message> {
    button::Button::new(content).style(primary_button_style)
}
