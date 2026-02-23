# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Development Methodology

This project strictly follows **Domain-Driven Design (DDD)**, **Clean Architecture (CA)**, **Test-Driven Development (TDD)**, and **Screaming Architecture**. All new code must adhere to these principles:

- **TDD**: Write tests first, then implement. Every domain service and entity must have corresponding tests.
- **Clean Architecture**: Dependencies point inward only (infrastructure → application → domain). The domain layer has zero external dependencies (ndarray is permitted as a numerical primitive).
- **DDD**: Business logic lives in domain entities and services, not in infrastructure or application layers.
- **Screaming Architecture**: Code is organized by feature slice (`detection/`, `blurring/`, `video/`), each containing `domain/` and `infrastructure/` sub-packages.

## Monorepo Structure

This is a Cargo workspace with three crates:

- **`crates/core`** — Core face detection and blurring library
- **`crates/cli`** — CLI binary (`video-blur`)
- **`crates/desktop`** — Desktop GUI binary (`video-blur-desktop`)

## Build & Run Commands

```bash
cargo build                                  # Build all crates
cargo test                                   # Run all tests
cargo test -p video-blur-core                # Run only core tests
cargo test -p video-blur-core -- region      # Run tests matching "region"
cargo run -p video-blur-cli -- input.mp4 output.mp4  # Run CLI
cargo run -p video-blur-desktop              # Run desktop GUI
cargo clippy --all-targets                   # Lint
cargo fmt --check                            # Check formatting
```

## Architecture

```
crates/
├── core/
│   └── src/
│       ├── lib.rs
│       ├── shared/                          # Cross-cutting domain entities
│       │   ├── frame.rs                     # Frame entity
│       │   ├── region.rs                    # Region entity (immutable struct)
│       │   ├── video_metadata.rs            # VideoMetadata entity
│       │   └── model_resolver.rs            # Model path resolution + download
│       ├── detection/                       # Face detection feature slice
│       │   ├── domain/                      # Pure business logic (traits + entities)
│       │   └── infrastructure/              # External library implementations
│       ├── blurring/                        # Frame blurring feature slice
│       │   ├── domain/
│       │   └── infrastructure/
│       ├── video/                           # Video I/O feature slice
│       │   ├── domain/
│       │   └── infrastructure/
│       └── pipeline/                        # Application layer (use case orchestration)
├── cli/
│   └── src/
│       └── main.rs                          # CLI entry point + arg parsing
└── desktop/
    └── src/
        ├── main.rs                          # App entry point
        ├── app.rs                           # Top-level iced Application
        ├── tabs/                            # UI tab implementations
        └── widgets/                         # Custom widgets
```

**Dependency rule**: `infrastructure` depends on `domain`. `pipeline` (application) depends only on `domain` traits, never on `infrastructure`. `domain` depends on nothing external (ndarray is permitted).

**Package boundary**: The desktop and CLI crates depend on core via `video-blur-core`. They import use cases and infrastructure implementations to wire them together.

## Code Conventions

- Rust 2021 edition, MSRV 1.75
- Type hints on all public functions
- Immutable structs for domain entities (no `&mut self` on entities, return new instances)
- `&mut Frame` for blur operations (zero-copy, borrow checker ensures safety)
- RGB pixel format internally; convert at I/O boundaries only
- `thiserror` for error types
- Traits for all domain interfaces (FaceDetector, FrameBlurrer, VideoReader, VideoWriter, etc.)
- `#[derive(Clone, Debug, PartialEq)]` on domain entities

## Testing Conventions

- TDD: write tests before implementation
- `rstest` for parametrized tests (equivalent to pytest.mark.parametrize)
- `approx` crate for float comparisons (equivalent to pytest.approx)
- Stub/fake trait implementations for isolation (no mocking library)
- Infrastructure tests that require models or network marked with `#[ignore]`
- Test helpers as module-level functions (e.g., `fn region(x, y, w, h) -> Region`)

## Key Domain Constants (from Appendix A of PLAN.md)

These values are contractual and must be preserved exactly:

| Constant | Value | Location |
|----------|-------|----------|
| IoU dedup threshold | 0.3 | Region::deduplicate |
| Landmark weights | [2, 2, 3, 1, 1] | FaceLandmarks |
| Padding | 0.4 | FaceRegionBuilder |
| Min width ratio | 0.8 | FaceRegionBuilder |
| EMA alpha | 0.6 | RegionSmoother |
| Edge fraction | 0.25 | RegionMerger |
| Default lookahead | 5 | BlurFacesUseCase |
| Thread queue capacity | 4 | BlurFacesUseCase |
| Preview crop size | 256 | PreviewFacesUseCase |

Refer to `PLAN.md` Appendix A for complete formulas and behavioral rules.
