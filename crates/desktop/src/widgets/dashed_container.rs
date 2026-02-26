use std::time::Instant;

use iced::advanced::graphics::geometry;
use iced::advanced::layout;
use iced::advanced::renderer;
use iced::advanced::widget::tree::{self, Tree};
use iced::advanced::widget::Widget;
use iced::advanced::{Clipboard, Layout, Renderer as _, Shell};
use iced::border::Border;
use iced::widget::canvas::{self, Frame, Path, Stroke};
use iced::{
    alignment, Color, Element, Event, Length, Padding, Point, Rectangle, Renderer, Size, Theme,
};

const ANIMATION_SECS: f32 = 0.2;

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
    hover_style: Option<DashedBorderStyle>,
    hovered: bool,
    padding: Padding,
    width: Length,
}

#[derive(Debug)]
struct AnimState {
    hover_amount: f32,
    target: f32,
    last_tick: Instant,
}

impl Default for AnimState {
    fn default() -> Self {
        Self {
            hover_amount: 0.0,
            target: 0.0,
            last_tick: Instant::now(),
        }
    }
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
            hover_style: None,
            hovered: false,
            padding: padding.into(),
            width: Length::Fill,
        }
    }

    pub fn hover_style(mut self, style: DashedBorderStyle, hovered: bool) -> Self {
        self.hover_style = Some(style);
        self.hovered = hovered;
        self
    }
}

impl<Message> Widget<Message, Theme, Renderer> for DashedContainer<'_, Message> {
    fn tag(&self) -> tree::Tag {
        tree::Tag::of::<AnimState>()
    }

    fn state(&self) -> tree::State {
        tree::State::new(AnimState::default())
    }

    fn children(&self) -> Vec<Tree> {
        vec![Tree::new(&self.content)]
    }

    fn diff(&self, tree: &mut Tree) {
        if tree.children.len() == 1 {
            tree.children[0].diff(&self.content);
        } else {
            tree.children = vec![Tree::new(&self.content)];
        }
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
                self.content.as_widget_mut().layout(
                    &mut tree.children[0],
                    renderer,
                    &limits.loose(),
                )
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
            &mut tree.children[0],
            event,
            layout.children().next().unwrap(),
            cursor,
            renderer,
            clipboard,
            shell,
            viewport,
        );

        if self.hover_style.is_some() {
            let Event::Window(iced::window::Event::RedrawRequested(now)) = event else {
                return;
            };

            let state = tree.state.downcast_mut::<AnimState>();
            state.target = if self.hovered { 1.0 } else { 0.0 };

            if (state.hover_amount - state.target).abs() > 0.001 {
                let dt = now.duration_since(state.last_tick).as_secs_f32();
                let speed = 1.0 / ANIMATION_SECS;
                if state.hover_amount < state.target {
                    state.hover_amount = (state.hover_amount + speed * dt).min(state.target);
                } else {
                    state.hover_amount = (state.hover_amount - speed * dt).max(state.target);
                }
                shell.request_redraw();
            } else {
                state.hover_amount = state.target;
            }

            state.last_tick = *now;
        }
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
            &tree.children[0],
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
            let s = self.effective_style(tree);

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

            self.content.as_widget().draw(
                &tree.children[0],
                renderer,
                theme,
                renderer_style,
                layout.children().next().unwrap(),
                cursor,
                &clipped_viewport,
            );

            let mut frame = Frame::new(renderer, bounds.size());

            let inset = s.border_width / 2.0;
            let top_left = Point::new(inset, inset);
            let rect_size = Size::new(
                bounds.width - s.border_width,
                bounds.height - s.border_width,
            );

            let border_path = Path::rounded_rectangle(top_left, rect_size, s.corner_radius.into());
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

            renderer.with_translation(iced::Vector::new(bounds.x, bounds.y), |renderer| {
                geometry::Renderer::draw_geometry(renderer, geom);
            });
        }
    }
}

impl<Message> DashedContainer<'_, Message> {
    fn effective_style(&self, tree: &Tree) -> DashedBorderStyle {
        match self.hover_style {
            Some(hover) => {
                let t = tree.state.downcast_ref::<AnimState>().hover_amount;
                DashedBorderStyle {
                    border_color: lerp_color(self.style.border_color, hover.border_color, t),
                    border_width: self.style.border_width,
                    dash_length: self.style.dash_length,
                    gap_length: self.style.gap_length,
                    corner_radius: self.style.corner_radius,
                    background: lerp_color(self.style.background, hover.background, t),
                }
            }
            None => self.style,
        }
    }
}

fn lerp_color(from: Color, to: Color, t: f32) -> Color {
    Color {
        r: from.r + (to.r - from.r) * t,
        g: from.g + (to.g - from.g) * t,
        b: from.b + (to.b - from.b) * t,
        a: from.a + (to.a - from.a) * t,
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
) -> DashedContainer<'a, Message> {
    DashedContainer::new(style, padding, content)
}
