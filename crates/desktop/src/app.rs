use iced::widget::{button, column, container, row, scrollable, text};
use iced::{Element, Length, Task, Theme};

use crate::tabs;

const WEBSITE_URL: &str = "https://www.neutrinographics.com/";

// ---------------------------------------------------------------------------
// Tab enum
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Main,
    Settings,
    Accessibility,
    Privacy,
    About,
}

impl Tab {
    const ALL: &[Tab] = &[
        Tab::Main,
        Tab::Settings,
        Tab::Accessibility,
        Tab::Privacy,
        Tab::About,
    ];

    fn label(self) -> &'static str {
        match self {
            Tab::Main => "Main",
            Tab::Settings => "Settings",
            Tab::Accessibility => "Accessibility",
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
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

pub struct App {
    active_tab: Tab,
}

impl App {
    pub fn new() -> (Self, Task<Message>) {
        (
            Self {
                active_tab: Tab::Main,
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
        }
        Task::none()
    }

    pub fn view(&self) -> Element<'_, Message> {
        // Tab bar
        let tab_bar = row(Tab::ALL
            .iter()
            .map(|&tab| {
                let label = text(tab.label()).size(13);
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
        let content: Element<Message> = match self.active_tab {
            Tab::Main => tabs::main_tab::view(),
            Tab::Settings => tabs::settings_tab::view(),
            Tab::Accessibility => tabs::accessibility_tab::view(),
            Tab::Privacy => tabs::privacy_tab::view(),
            Tab::About => tabs::about_tab::view(),
        };

        let tab_content = container(scrollable(content).height(Length::Fill))
            .padding(16)
            .height(Length::Fill);

        // Footer
        let footer = container(
            button(text("neutrinographics.com").size(11))
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
        Theme::Dark
    }
}
