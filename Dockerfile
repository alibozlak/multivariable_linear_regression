# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Builder: compiles the release binary.
#
# This stage and the runtime stage below are both Debian bookworm on purpose.
# The binary links dynamically against the builder's glibc, so pairing it with
# a runtime from a different Debian release would produce an image that builds
# fine and then fails to start.
# ---------------------------------------------------------------------------
FROM rust:1.94-slim-bookworm AS builder

WORKDIR /build

COPY Cargo.toml Cargo.lock ./
COPY src ./src

# The two cache mounts keep the crates.io registry and the target directory
# between builds, so editing src/ recompiles this crate alone rather than its
# dependencies as well. Neither mount becomes part of the image, which is why
# the finished binary is copied out to a path the next stage can reach.
#
# --release is not optional here: the training loop makes a million passes over
# the data set, and a debug build is an order of magnitude slower.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/build/target \
    cargo build --release && \
    cp target/release/multivariable_linear_regression /usr/local/bin/

# ---------------------------------------------------------------------------
# Runtime: carries the binary and nothing else — no Rust toolchain, no source,
# no build cache.
# ---------------------------------------------------------------------------
FROM debian:bookworm-slim AS runtime

# Training neither writes files nor opens sockets, so it has no reason to run
# as root.
RUN useradd --create-home --user-group app
USER app
WORKDIR /home/app

COPY --from=builder /usr/local/bin/multivariable_linear_regression /usr/local/bin/

# The binary trains on the bundled data set and prints the cost before and
# after, so this container runs to completion instead of serving anything.
CMD ["multivariable_linear_regression"]
