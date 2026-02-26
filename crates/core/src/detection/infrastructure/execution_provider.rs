/// Return the preferred ONNX execution providers for the current platform.
///
/// Falls back to CPU if the platform-specific provider is unavailable.
pub fn preferred_execution_providers() -> Vec<ort::execution_providers::ExecutionProviderDispatch> {
    #[cfg(target_os = "macos")]
    {
        vec![ort::execution_providers::CoreMLExecutionProvider::default().build()]
    }
    #[cfg(target_os = "windows")]
    {
        vec![ort::execution_providers::DirectMLExecutionProvider::default().build()]
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        vec![]
    }
}
