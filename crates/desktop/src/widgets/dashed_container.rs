use iced::advanced::graphics::geometry;
use iced::advanced::layout;
use iced::advanced::renderer;
use iced::advanced::widget::tree::Tree;
use iced::advanced::widget::Widget;
use iced::advanced::{Clipboard, Layout, Renderer as _, Shell};
use iced::border::Border;
use iced::widget::canvas::{self, Frame, Path, Stroke};
use iced::{
    alignment, Color, Element, Event, Length, Padding, Point, Rectangle, Renderer, Size, Theme,
};

/// Configuration for the dashed border appearance.
#[derive(Debug, Clone, Copy)]
pub struct DashedBorderStyle {
    pub border_color: Color,
    pub border_width: f32,
    pub dash_length: f32,
    pub gap_length: f32,
    pub corner_radius: f32,
    pub background: Color,
}

/// A container widget that draws a dashed rounded-rectangle border around its child.
pub struct DashedContainer<'a, Message> {
    content: Element<'a, Message>,
    style: DashedBorderStyle,
    padding: Padding,
    width: Length,
}

impl<'a, Message> DashedContainer<'a, Message> {
    pub fn new(
        style: DashedBorderStyle,
        padding: impl Into<Padding>,
        content: impl Into<Element<'a, Message>>,
    ) -> Self {
        Self {
            content: content.into(),
            style,
            padding: padding.into(),
            width: Length::Fill,
        }
    }
}

impl<Message> Widget<Message, Theme, Renderer> for DashedContainer<'_, Message> {
    fn tag(&self) -> iced::advanced::widget::tree::Tag {
        self.content.as_widget().tag()
    }

    fn state(&self) -> iced::advanced::widget::tree::State {
        self.content.as_widget().state()
    }

    fn children(&self) -> Vec<Tree> {
        self.content.as_widget().children()
    }

    fn diff(&self, tree: &mut Tree) {
        self.content.as_widget().diff(tree);
    }

    fn size(&self) -> Size<Length> {
        Size {
            width: self.width,
            height: Length::Shrink,
        }
    }

    fn layout(
        &mut self,
        tree: &mut Tree,
        renderer: &Renderer,
        limits: &layout::Limits,
    ) -> layout::Node {
        layout::positioned(
            limits,
            self.width,
            Length::Shrink,
            self.padding,
            |limits| {
                self.content
                    .as_widget_mut()
                    .layout(tree, renderer, &limits.loose())
            },
            |content, size| {
                content.align(
                    alignment::Alignment::from(alignment::Horizontal::Center),
                    alignment::Alignment::from(alignment::Vertical::Top),
                    size,
                )
            },
        )
    }

    fn update(
        &mut self,
        tree: &mut Tree,
        event: &Event,
        layout: Layout<'_>,
        cursor: iced::mouse::Cursor,
        renderer: &Renderer,
        clipboard: &mut dyn Clipboard,
        shell: &mut Shell<'_, Message>,
        viewport: &Rectangle,
    ) {
        self.content.as_widget_mut().update(
            tree,
            event,
            layout.children().next().unwrap(),
            cursor,
            renderer,
            clipboard,
            shell,
            viewport,
        );
    }

    fn mouse_interaction(
        &self,
        tree: &Tree,
        layout: Layout<'_>,
        cursor: iced::mouse::Cursor,
        viewport: &Rectangle,
        renderer: &Renderer,
    ) -> iced::mouse::Interaction {
        self.content.as_widget().mouse_interaction(
            tree,
            layout.children().next().unwrap(),
            cursor,
            viewport,
            renderer,
        )
    }

    fn draw(
        &self,
        tree: &Tree,
        renderer: &mut Renderer,
        theme: &Theme,
        renderer_style: &renderer::Style,
        layout: Layout<'_>,
        cursor: iced::mouse::Cursor,
        viewport: &Rectangle,
    ) {
        let bounds = layout.bounds();

        if let Some(clipped_viewport) = bounds.intersection(viewport) {
            let s = &self.style;

            // Draw background using fill_quad (same render pass as buttons)
            renderer.fill_quad(
                renderer::Quad {
                    bounds,
                    border: Border {
                        radius: s.corner_radius.into(),
                        ..Border::default()
                    },
                    ..renderer::Quad::default()
                },
                s.background,
            );

            // Draw child content
            self.content.as_widget().draw(
                tree,
                renderer,
                theme,
                renderer_style,
                layout.children().next().unwrap(),
                cursor,
                &clipped_viewport,
            );

            // Draw dashed border on top (only at edges, won't overlap inner content)
            let mut frame = Frame::new(renderer, bounds.size());

            let inset = s.border_width / 2.0;
            let top_left = Point::new(inset, inset);
            let rect_size = Size::new(
                bounds.width - s.border_width,
                bounds.height - s.border_width,
            );

            let border_path =
                Path::rounded_rectangle(top_left, rect_size, s.corner_radius.into());
            let dash_pattern = [s.dash_length, s.gap_length];
            frame.stroke(
                &border_path,
                Stroke {
                    style: canvas::Style::Solid(s.border_color),
                    width: s.border_width,
                    line_cap: canvas::LineCap::Round,
                    line_dash: canvas::LineDash {
                        segments: &dash_pattern,
                        offset: 0,
                    },
                    ..Stroke::default()
                },
            );

            let geom = frame.into_geometry();

            renderer.with_translation(
                iced::Vector::new(bounds.x, bounds.y),
                |renderer| {
                    geometry::Renderer::draw_geometry(renderer, geom);
                },
            );
        }
    }
}

impl<'a, Message: 'a> From<DashedContainer<'a, Message>> for Element<'a, Message> {
    fn from(container: DashedContainer<'a, Message>) -> Self {
        Element::new(container)
    }
}

/// Creates a dashed-border container element with the given content and style.
pub fn dashed_container<'a, Message: 'a>(
    style: DashedBorderStyle,
    padding: impl Into<Padding>,
    content: impl Into<Element<'a, Message>>,
) -> Element<'a, Message> {
    DashedContainer::new(style, padding, content).into()
}
