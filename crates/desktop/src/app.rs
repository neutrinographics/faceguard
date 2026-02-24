use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crossbeam_channel::Receiver;
use iced::widget::{button, column, container, row, scrollable, text, Space};
use iced::{Element, Length, Subscription, Task, Theme};

use crate::settings::{Appearance, BlurShape, Settings};
use crate::tabs;
use crate::theme;
use crate::widgets::faces_well::FacesWellState;
use crate::workers::blur_worker::{self, BlurParams, WorkerMessage};
use crate::workers::preview_worker::{self, PreviewMessage, PreviewParams};
use video_blur_core::shared::region::Region;

const WEBSITE_URL: &str = "https://www.neutrinographics.com/";

// ---------------------------------------------------------------------------
// Tab enum
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Blur,
    Settings,
    About,
}

impl Tab {
    const ALL: &[Tab] = &[Tab::Blur, Tab::Settings, Tab::About];

    fn label(self) -> &'static str {
        match self {
            Tab::Blur => "Blur",
            Tab::Settings => "Settings",
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
    Scanning(usize, usize),
    Previewed,
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
    RunPreview,
    RunBlur,
    CancelWork,
    WorkerTick,
    ShowInFolder,
    StartOver,
    ToggleFace(u32),
    ToggleGroup(usize),
    GroupFacesToggled(bool),
    BlurShapeChanged(BlurShape),
    ConfidenceChanged(u32),
    BlurStrengthChanged(u32),
    LookaheadChanged(u32),
    RestoreDefaults,
    AppearanceChanged(Appearance),
    HighContrastChanged(bool),
    FontScaleChanged(f32),
    PollSystemTheme,
    FileDropped(PathBuf),
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
    pub faces_well: FacesWellState,
    /// Cached detection results from preview pass (for reuse in blur).
    detection_cache: Option<HashMap<usize, Vec<Region>>>,
    preview_rx: Option<Receiver<PreviewMessage>>,
    worker_rx: Option<Receiver<WorkerMessage>>,
    worker_cancel: Option<Arc<AtomicBool>>,
}

impl App {
    pub fn new() -> (Self, Task<Message>) {
        (
            Self {
                active_tab: Tab::Blur,
                settings: Settings::load(),
                input_path: None,
                output_path: None,
                processing: ProcessingState::Idle,
                faces_well: FacesWellState::new(),
                detection_cache: None,
                preview_rx: None,
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
                // Clear previous preview data
                self.faces_well.clear();
                self.detection_cache = None;
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
            Message::RunPreview => {
                if let Some(input) = self.input_path.clone() {
                    let params = PreviewParams {
                        input_path: input,
                        confidence: self.settings.confidence,
                    };
                    let (rx, cancel) = preview_worker::spawn(params);
                    self.preview_rx = Some(rx);
                    self.worker_cancel = Some(cancel);
                    self.processing = ProcessingState::Preparing;
                }
            }
            Message::RunBlur => {
                if let (Some(input), Some(output)) =
                    (self.input_path.clone(), self.output_path.clone())
                {
                    let blur_ids = self.faces_well.get_selected_ids();
                    let params = BlurParams {
                        input_path: input,
                        output_path: output,
                        blur_shape: self.settings.blur_shape,
                        confidence: self.settings.confidence,
                        blur_strength: self.settings.blur_strength,
                        lookahead: self.settings.lookahead,
                        detection_cache: self.detection_cache.clone(),
                        blur_ids,
                    };
                    let (rx, cancel) = blur_worker::spawn(params);
                    self.worker_rx = Some(rx);
                    self.worker_cancel = Some(cancel);
                    self.processing = ProcessingState::Preparing;
                }
            }
            Message::CancelWork => {
                if let Some(ref cancel) = self.worker_cancel {
                    cancel.store(true, Ordering::Relaxed);
                }
            }
            Message::WorkerTick => {
                // Collect preview messages
                let preview_msgs: Vec<_> = self
                    .preview_rx
                    .as_ref()
                    .map(|rx| rx.try_iter().collect())
                    .unwrap_or_default();

                for msg in preview_msgs {
                    match msg {
                        PreviewMessage::DownloadProgress(dl, total) => {
                            self.processing = ProcessingState::Downloading(dl, total);
                        }
                        PreviewMessage::ScanProgress(current, total) => {
                            self.processing = ProcessingState::Scanning(current, total);
                        }
                        PreviewMessage::Complete(result) => {
                            self.faces_well
                                .populate(result.crops, result.groups, result.temp_dir);
                            self.detection_cache = Some(result.detection_cache);
                            self.processing = ProcessingState::Previewed;
                            self.preview_rx = None;
                            self.worker_cancel = None;
                        }
                        PreviewMessage::Error(e) => {
                            self.processing = ProcessingState::Error(e);
                            self.preview_rx = None;
                            self.worker_cancel = None;
                        }
                        PreviewMessage::Cancelled => {
                            self.processing = ProcessingState::Idle;
                            self.preview_rx = None;
                            self.worker_cancel = None;
                        }
                    }
                }

                // Collect blur messages
                let blur_msgs: Vec<_> = self
                    .worker_rx
                    .as_ref()
                    .map(|rx| rx.try_iter().collect())
                    .unwrap_or_default();

                for msg in blur_msgs {
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
            Message::ToggleFace(track_id) => {
                self.faces_well.toggle_face(track_id);
            }
            Message::ToggleGroup(group_idx) => {
                self.faces_well.toggle_group(group_idx);
            }
            Message::GroupFacesToggled(enabled) => {
                self.faces_well.group_faces = enabled;
            }
            Message::ShowInFolder => {
                if let Some(ref output) = self.output_path {
                    if let Some(parent) = output.parent() {
                        let _ = open::that(parent);
                    }
                }
            }
            Message::StartOver => {
                self.processing = ProcessingState::Idle;
                self.input_path = None;
                self.output_path = None;
                self.faces_well.clear();
                self.detection_cache = None;
            }
            Message::BlurShapeChanged(shape) => {
                self.settings.blur_shape = shape;
                self.settings.save();
            }
            Message::ConfidenceChanged(val) => {
                self.settings.confidence = val;
                self.settings.save();
                self.invalidate_detection();
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
                let detection_changed = self.settings.confidence != defaults.confidence;
                self.settings.blur_shape = defaults.blur_shape;
                self.settings.confidence = defaults.confidence;
                self.settings.blur_strength = defaults.blur_strength;
                self.settings.lookahead = defaults.lookahead;
                self.settings.appearance = defaults.appearance;
                self.settings.high_contrast = defaults.high_contrast;
                self.settings.font_scale = defaults.font_scale;
                self.settings.save();
                if detection_changed {
                    self.invalidate_detection();
                }
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
            Message::FileDropped(path) => {
                let ext = path
                    .extension()
                    .and_then(|e| e.to_str())
                    .map(|e| e.to_lowercase())
                    .unwrap_or_default();
                let supported = [
                    "mp4", "avi", "mov", "mkv", "jpg", "jpeg", "png", "bmp", "tiff", "webp",
                ];
                if supported.contains(&ext.as_str()) {
                    return self.update(Message::InputSelected(Some(path)));
                }
            }
        }
        Task::none()
    }

    pub fn view(&self) -> Element<'_, Message> {
        let fs = self.settings.font_scale;
        let current_theme = self.theme();
        let palette = current_theme.palette();
        let muted = theme::muted_color(&current_theme);

        // Underline-style tab bar
        let tab_bar = container(
            row(Tab::ALL
                .iter()
                .map(|&tab| {
                    let is_active = tab == self.active_tab;
                    let label_color = if is_active { palette.primary } else { muted };
                    let label = text(tab.label())
                        .size(scaled(13.0, fs))
                        .color(label_color);
                    let btn = button(label)
                        .on_press(Message::TabSelected(tab))
                        .padding([12, 20])
                        .style(button::text);

                    let bar: Element<'_, Message> = if is_active {
                        container(Space::new().height(0))
                            .width(Length::Fill)
                            .height(2)
                            .style(move |_theme: &Theme| container::Style {
                                background: Some(palette.primary.into()),
                                ..Default::default()
                            })
                            .into()
                    } else {
                        Space::new().height(2).into()
                    };

                    column![btn, bar].align_x(iced::Alignment::Center).into()
                })
                .collect::<Vec<_>>())
            .spacing(0),
        )
        .padding([0, 8]);

        // Tab content
        let content: Element<'_, Message> = match self.active_tab {
            Tab::Blur => {
                tabs::main_tab::view(
                    fs,
                    self.input_path.as_deref(),
                    self.output_path.as_deref(),
                    &self.processing,
                    &self.faces_well,
                    &current_theme,
                )
            }
            Tab::Settings => tabs::settings_tab::view(&self.settings),
            Tab::About => tabs::about_tab::view(fs),
        };

        let tab_content = container(scrollable(content).height(Length::Fill))
            .padding(24)
            .height(Length::Fill);

        column![tab_bar, tab_content]
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

        if self.worker_rx.is_some() || self.preview_rx.is_some() {
            subs.push(iced::time::every(Duration::from_millis(50)).map(|_| Message::WorkerTick));
        }

        // Listen for file drops onto the window
        subs.push(iced::event::listen_with(|event, _status, _id| {
            if let iced::Event::Window(iced::window::Event::FileDropped(path)) = event {
                Some(Message::FileDropped(path))
            } else {
                None
            }
        }));

        Subscription::batch(subs)
    }
}

impl App {
    /// Invalidate cached detection when settings change.
    fn invalidate_detection(&mut self) {
        if self.detection_cache.is_some() {
            self.detection_cache = None;
            self.faces_well.clear();
            if matches!(self.processing, ProcessingState::Previewed) {
                self.processing = ProcessingState::Idle;
            }
        }
    }
}

/// Scale a base font size by the user's font_scale setting.
pub fn scaled(base: f32, font_scale: f32) -> f32 {
    (base * font_scale).round()
}
