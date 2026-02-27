# Workaround for Apple Clang 17+ which does not support -mcpu=native.
# whisper-rs-sys / ggml uses GGML_NATIVE which passes -mcpu=native to the compiler.
# See: https://github.com/tazz4843/whisper-rs/issues/XXX
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
