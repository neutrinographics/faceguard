use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crossbeam_channel::Receiver;
use iced::widget::{button, column, container, row, scrollable, text};
use iced::{Element, Length, Subscription, Task, Theme};

use crate::settings::{Appearance, BlurShape, Detector, Settings};
use crate::tabs;
use crate::theme;
use crate::workers::blur_worker::{self, BlurParams, WorkerMessage};

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
// Processing state
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum ProcessingState {
    Idle,
    Preparing,
    Downloading(u64, u64),
    Blurring(usize, usize),
    Complete,
    Error(String),
}

// ---------------------------------------------------------------------------
// Message
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum Message {
    TabSelected(Tab),
    OpenWebsite,
    SelectInput,
    InputSelected(Option<PathBuf>),
    SelectOutput,
    OutputSelected(Option<PathBuf>),
    RunBlur,
    CancelBlur,
    WorkerTick,
    DetectorChanged(Detector),
    BlurShapeChanged(BlurShape),
    ConfidenceChanged(u32),
    BlurStrengthChanged(u32),
    LookaheadChanged(u32),
    RestoreDefaults,
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
    pub input_path: Option<PathBuf>,
    pub output_path: Option<PathBuf>,
    pub processing: ProcessingState,
    worker_rx: Option<Receiver<WorkerMessage>>,
    worker_cancel: Option<Arc<AtomicBool>>,
}

impl App {
    pub fn new() -> (Self, Task<Message>) {
        (
            Self {
                active_tab: Tab::Main,
                settings: Settings::load(),
                input_path: None,
                output_path: None,
                processing: ProcessingState::Idle,
                worker_rx: None,
                worker_cancel: None,
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
            Message::SelectInput => {
                return Task::perform(
                    async {
                        rfd::AsyncFileDialog::new()
                            .set_title("Select input file")
                            .add_filter(
                                "Media Files",
                                &[
                                    "mp4", "avi", "mov", "mkv", "jpg", "jpeg", "png", "bmp",
                                    "tiff", "webp",
                                ],
                            )
                            .pick_file()
                            .await
                            .map(|h| h.path().to_path_buf())
                    },
                    Message::InputSelected,
                );
            }
            Message::InputSelected(Some(path)) => {
                let stem = path
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();
                let ext = path
                    .extension()
                    .map(|e| format!(".{}", e.to_string_lossy()))
                    .unwrap_or_default();
                let output = path.with_file_name(format!("{stem}_blurred{ext}"));
                self.input_path = Some(path);
                self.output_path = Some(output);
                self.processing = ProcessingState::Idle;
            }
            Message::InputSelected(None) => {}
            Message::SelectOutput => {
                let start_dir = self
                    .output_path
                    .as_ref()
                    .and_then(|p| p.parent().map(|d| d.to_path_buf()));
                let start_name = self
                    .output_path
                    .as_ref()
                    .and_then(|p| p.file_name().map(|n| n.to_string_lossy().to_string()));
                return Task::perform(
                    async move {
                        let mut dialog = rfd::AsyncFileDialog::new()
                            .set_title("Save output as")
                            .add_filter(
                                "Media Files",
                                &[
                                    "mp4", "avi", "mov", "mkv", "jpg", "jpeg", "png", "bmp",
                                    "tiff", "webp",
                                ],
                            );
                        if let Some(dir) = start_dir {
                            dialog = dialog.set_directory(dir);
                        }
                        if let Some(name) = start_name {
                            dialog = dialog.set_file_name(name);
                        }
                        dialog.save_file().await.map(|h| h.path().to_path_buf())
                    },
                    Message::OutputSelected,
                );
            }
            Message::OutputSelected(Some(path)) => {
                self.output_path = Some(path);
            }
            Message::OutputSelected(None) => {}
            Message::RunBlur => {
                if let (Some(input), Some(output)) =
                    (self.input_path.clone(), self.output_path.clone())
                {
                    let params = BlurParams {
                        input_path: input,
                        output_path: output,
                        detector: self.settings.detector,
                        blur_shape: self.settings.blur_shape,
                        confidence: self.settings.confidence,
                        blur_strength: self.settings.blur_strength,
                        lookahead: self.settings.lookahead,
                    };
                    let (rx, cancel) = blur_worker::spawn(params);
                    self.worker_rx = Some(rx);
                    self.worker_cancel = Some(cancel);
                    self.processing = ProcessingState::Preparing;
                }
            }
            Message::CancelBlur => {
                if let Some(ref cancel) = self.worker_cancel {
                    cancel.store(true, Ordering::Relaxed);
                }
            }
            Message::WorkerTick => {
                // Collect messages first to avoid borrow conflict
                let msgs: Vec<_> = self
                    .worker_rx
                    .as_ref()
                    .map(|rx| rx.try_iter().collect())
                    .unwrap_or_default();

                for msg in msgs {
                    match msg {
                        WorkerMessage::DownloadProgress(dl, total) => {
                            self.processing = ProcessingState::Downloading(dl, total);
                        }
                        WorkerMessage::BlurProgress(current, total) => {
                            self.processing = ProcessingState::Blurring(current, total);
                        }
                        WorkerMessage::Complete => {
                            self.processing = ProcessingState::Complete;
                            self.worker_rx = None;
                            self.worker_cancel = None;
                        }
                        WorkerMessage::Error(e) => {
                            self.processing = ProcessingState::Error(e);
                            self.worker_rx = None;
                            self.worker_cancel = None;
                        }
                        WorkerMessage::Cancelled => {
                            self.processing = ProcessingState::Idle;
                            self.worker_rx = None;
                            self.worker_cancel = None;
                        }
                    }
                }
            }
            Message::DetectorChanged(detector) => {
                self.settings.detector = detector;
                self.settings.save();
            }
            Message::BlurShapeChanged(shape) => {
                self.settings.blur_shape = shape;
                self.settings.save();
            }
            Message::ConfidenceChanged(val) => {
                self.settings.confidence = val;
                self.settings.save();
            }
            Message::BlurStrengthChanged(val) => {
                self.settings.blur_strength = if val % 2 == 0 { val + 1 } else { val };
                self.settings.save();
            }
            Message::LookaheadChanged(val) => {
                self.settings.lookahead = val;
                self.settings.save();
            }
            Message::RestoreDefaults => {
                let defaults = Settings::default();
                self.settings.detector = defaults.detector;
                self.settings.blur_shape = defaults.blur_shape;
                self.settings.confidence = defaults.confidence;
                self.settings.blur_strength = defaults.blur_strength;
                self.settings.lookahead = defaults.lookahead;
                self.settings.save();
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
            Message::PollSystemTheme => {}
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
            Tab::Main => tabs::main_tab::view(
                fs,
                self.input_path.as_deref(),
                self.output_path.as_deref(),
                &self.processing,
            ),
            Tab::Settings => tabs::settings_tab::view(&self.settings),
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
        let mut subs = vec![];

        if self.settings.appearance == Appearance::System {
            subs.push(iced::time::every(Duration::from_secs(2)).map(|_| Message::PollSystemTheme));
        }

        if self.worker_rx.is_some() {
            subs.push(iced::time::every(Duration::from_millis(50)).map(|_| Message::WorkerTick));
        }

        Subscription::batch(subs)
    }
}

/// Scale a base font size by the user's font_scale setting.
pub fn scaled(base: f32, font_scale: f32) -> f32 {
    (base * font_scale).round()
}
