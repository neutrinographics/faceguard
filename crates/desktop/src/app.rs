use std::time::Duration;

use iced::widget::{button, column, container, row, scrollable, text};
use iced::{Element, Length, Subscription, Task, Theme};

use crate::settings::{Appearance, Settings};
use crate::tabs;
use crate::theme;

const WEBSITE_URL: &str = "https://www.neutrinographics.com/";

// ---------------------------------------------------------------------------
// Tab enum
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Main,
    Settings,
    Appearance,
    Privacy,
    About,
}

impl Tab {
    const ALL: &[Tab] = &[
        Tab::Main,
        Tab::Settings,
        Tab::Appearance,
        Tab::Privacy,
        Tab::About,
    ];

    fn label(self) -> &'static str {
        match self {
            Tab::Main => "Main",
            Tab::Settings => "Settings",
            Tab::Appearance => "Appearance",
            Tab::Privacy => "Privacy",
            Tab::About => "About",
        }
    }
}

// ---------------------------------------------------------------------------
// Message
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum Message {
    TabSelected(Tab),
    OpenWebsite,
    AppearanceChanged(Appearance),
    HighContrastChanged(bool),
    FontScaleChanged(f32),
    PollSystemTheme,
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

pub struct App {
    active_tab: Tab,
    pub settings: Settings,
}

impl App {
    pub fn new() -> (Self, Task<Message>) {
        (
            Self {
                active_tab: Tab::Main,
                settings: Settings::load(),
            },
            Task::none(),
        )
    }

    pub fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::TabSelected(tab) => {
                self.active_tab = tab;
            }
            Message::OpenWebsite => {
                let _ = open::that(WEBSITE_URL);
            }
            Message::AppearanceChanged(appearance) => {
                self.settings.appearance = appearance;
                self.settings.save();
            }
            Message::HighContrastChanged(enabled) => {
                self.settings.high_contrast = enabled;
                self.settings.save();
            }
            Message::FontScaleChanged(scale) => {
                self.settings.font_scale = scale;
                self.settings.save();
            }
            Message::PollSystemTheme => {
                // Theme is resolved fresh in theme() on every render,
                // so just requesting a redraw is enough.
            }
        }
        Task::none()
    }

    pub fn view(&self) -> Element<'_, Message> {
        let fs = self.settings.font_scale;

        // Tab bar
        let tab_bar = row(Tab::ALL
            .iter()
            .map(|&tab| {
                let label = text(tab.label()).size(scaled(13.0, fs));
                let btn = button(label)
                    .on_press(Message::TabSelected(tab))
                    .padding([6, 14]);
                if tab == self.active_tab {
                    btn.style(button::primary).into()
                } else {
                    btn.style(button::text).into()
                }
            })
            .collect::<Vec<_>>())
        .spacing(2);

        // Tab content
        let content: Element<'_, Message> = match self.active_tab {
            Tab::Main => tabs::main_tab::view(fs),
            Tab::Settings => tabs::settings_tab::view(fs),
            Tab::Appearance => tabs::appearance_tab::view(&self.settings),
            Tab::Privacy => tabs::privacy_tab::view(fs),
            Tab::About => tabs::about_tab::view(fs),
        };

        let tab_content = container(scrollable(content).height(Length::Fill))
            .padding(16)
            .height(Length::Fill);

        // Footer
        let footer = container(
            button(text("neutrinographics.com").size(scaled(11.0, fs)))
                .on_press(Message::OpenWebsite)
                .style(button::text),
        )
        .width(Length::Fill)
        .center_x(Length::Fill)
        .padding([4, 0]);

        column![tab_bar, tab_content, footer]
            .spacing(0)
            .height(Length::Fill)
            .into()
    }

    pub fn theme(&self) -> Theme {
        theme::resolve_theme(self.settings.appearance, self.settings.high_contrast)
    }

    pub fn subscription(&self) -> Subscription<Message> {
        if self.settings.appearance == Appearance::System {
            iced::time::every(Duration::from_secs(2)).map(|_| Message::PollSystemTheme)
        } else {
            Subscription::none()
        }
    }
}

/// Scale a base font size by the user's font_scale setting.
pub fn scaled(base: f32, font_scale: f32) -> f32 {
    (base * font_scale).round()
}
