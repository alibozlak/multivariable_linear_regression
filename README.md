# Multivariable Linear Regression

Multivariable linear regression written from scratch in Rust — no ML crates and
no linear algebra library. The model itself runs on `std` alone; the only
dependencies in the project are `serde` and `serde_json`, and they are used
solely by the JSON converter, never by the model.

The goal of the project is not to get the best possible prediction, but to
implement gradient descent step by step: the cost function, its partial
derivatives derived by hand, and the simultaneous parameter update. The first
model is deliberately trained **without feature scaling**, so that the price you
pay for skipping that step becomes visible in the numbers.

The data set is a synthetic rental-price table: the square meters and the room
count of a flat go in, the monthly rent comes out.

## The math

The model function (hypothesis) with `n` features:

```
f(x) = a_1 * x_1 + a_2 * x_2 + ... + a_n * x_n + b
```

The cost function — mean squared error over `m` samples:

```
J(a_1, ..., a_n, b) = (1/m) * sum over i of ( f(x^(i)) - y^(i) )^2
```

There is no `1/2` factor in front of the mean, which is why the factor `2`
survives in both derivatives below.

Gradient descent repeats the following update, where `alpha` is the learning
rate:

```
a_j := a_j - alpha * dJ/da_j        for every j = 1..n
b   := b   - alpha * dJ/db
```

with the partial derivatives

```
dJ/da_j = (2/m) * sum over i of ( x_j^(i) * ( f(x^(i)) - y^(i) ) )
dJ/db   = (2/m) * sum over i of ( f(x^(i)) - y^(i) )
```

Both derivatives are worked out by hand from the limit definition of the
derivative, in the `math/` directory:

| File | Content |
| --- | --- |
| `math/001_dJ_daj_partial_derivative.pdf` | Derivation of `dJ/da_j` |
| `math/002_dJ_db_partial_derivative.pdf` | Derivation of `dJ/db` |

One detail that is easy to get wrong: the update has to be **simultaneous**. All
partial derivatives of an iteration must be computed from the same, still
unchanged coefficients. `train_model` therefore collects the new values in a
temporary vector and only copies them back once the whole iteration is done.

## Project structure

```
.
├── src
│   ├── main.rs                      # Entry point: hyperparameters, training, run log
│   ├── without_feature_scaling.rs   # The model: cost, derivatives, gradient descent
│   ├── json_converter.rs            # JSON <-> the vectors the model works with
│   └── dataset.rs                   # Synthetic rental data set (100 samples)
├── math
│   ├── 001_dJ_daj_partial_derivative.pdf
│   └── 002_dJ_db_partial_derivative.pdf
├── Cargo.toml
└── README.md
```

The coefficients are kept in a single vector of length `n + 1`, laid out as
`[a_1, ..., a_n, b]` — the last slot is the bias.

## Requirements

* Rust 1.85 or newer (the crate uses edition 2024)
* `serde` and `serde_json`, pulled in by Cargo — needed by the JSON converter only

`serde_json` is enabled with its `float_roundtrip` feature. Without it the
parser is allowed to be off by one ULP when reading a float back, so coefficients
written out and read in again would not be bit-identical to the ones training
produced.

## Running it

```bash
git clone https://github.com/alibozlak/multivariable_linear_regression.git
cd multivariable_linear_regression
cargo run --release
```

`--release` matters: the training loop performs a million passes over the whole
data set, and a debug build is an order of magnitude slower.

Output:

```
J (Before model training) = 1383285.4039156951

J (After model training) = 1365170.8112169602

Last coefficients: [375.1373157216981, -195.00662490954502, 1807.2879939508762]
```

## The data set

`src/dataset.rs` holds 100 samples produced with Gemini.

* Input (`x`): `[square meters, room count]` — roughly 40..185 m² and 1..5 rooms
* Output (`y`): monthly rent, roughly 18,000..75,000

The gap between the ranges of the two features is exactly the point. Square
meters move on a scale about forty times larger than the room count, so the
cost surface turns into a long, narrow valley: descent races down the steep
direction and crawls along the flat one.

## Training log

Every run does 1,000,000 iterations with `alpha = 0.000003`, and each one picks
up from the coefficients the previous run ended with:

| Run | Starting coefficients | J before | J after | Resulting coefficients |
| --- | --- | --- | --- | --- |
| 1 | `[0.0, 0.0, 0.0]` | 1,567,635,000 | 1,465,777 | `[382.428, -226.377, 1100.192]` |
| 2 | result of run 1 | 1,465,777 | 1,383,285 | `[378.423, -240.756, 1583.881]` |
| 3 | result of run 2 | 1,383,285 | 1,365,171 | `[375.137, -195.007, 1807.288]` |

Since `J` is a mean squared error, its square root is readable in the unit of
the price itself: `sqrt(1,365,171) ≈ 1,168`, so the model is off by about 1,168
per flat on average — against rents in the tens of thousands.

Three things worth reading out of this table:

1. **The learning rate cannot be raised.** With unscaled features, anything much
   larger than `0.000003` makes the cost diverge instead of fall. The small step
   is what forces the million iterations.
2. **It has not converged yet.** The bias is still climbing steadily
   (1100 → 1583 → 1807) and `J` is still falling. That slow drift along the flat
   direction of the valley is the textbook symptom of missing feature scaling.
3. **The room-count coefficient is negative.** That does not mean extra rooms
   make a flat cheaper — room count and square meters are strongly correlated in
   this data set, so the model explains nearly everything through the square
   meters and uses the second coefficient only as a correction term.

To continue training, copy the printed coefficients into the third argument of
`WithoutFeatureScaling::new` in `src/main.rs` and run it again.

## Using your own data

```rust
let inputs: Vec<Vec<f64>> = vec![
    vec![55.0, 1.0],
    vec![130.0, 4.0],
];
let outputs: Vec<f64> = vec![23000.0, 48500.0];

// One coefficient per feature, plus the bias: n + 1 in total.
let mut model = WithoutFeatureScaling::new(inputs, outputs, vec![0.0; 3]);

println!("J = {}", model.J());
let coefficients = model.train_model(0.000003, 1_000_000);
```

`new` panics if the number of outputs does not match the number of input
samples, or if the coefficient vector is not `n + 1` long.

## JSON input and output

`src/json_converter.rs` bridges JSON and the vectors the model works with, so a
training set can come from a file or a request body instead of being compiled in.

A training set goes in as:

```json
{
  "inputs":  [[55.0, 1.0], [130.0, 4.0]],
  "outputs": [23000.0, 48500.0]
}
```

```rust
let (inputs, outputs) = json_converter::training_data_from_json(&body)?;
let mut model = WithoutFeatureScaling::new(inputs, outputs, vec![0.0; 3]);

let last_coefficients = model.train_model(0.000003, 1_000_000);
let json = json_converter::coefficients_to_json(&last_coefficients)?;
// {"last_coefficients":[375.1373157216981,-195.00662490954502,1807.2879939508762]}
```

The converter validates before handing anything to the model and returns a
`JsonConverterError` instead of panicking: it rejects an empty data set, a
mismatch between the number of samples and the number of outputs, and samples
whose feature counts differ from each other. That last check is one the model's
own `validate_data_set` does not perform, since it reads the feature count from
the first sample only. This matters when the JSON arrives from outside — a bad
payload becomes an error you can map to a 400 rather than a panic.

* [ ] A model **with** feature scaling, to compare convergence speed against the
      current one
* [ ] Vectorized (SIMD) prediction — the branch in `f()` is already reserved for
      it, both sides are identical for now
* [ ] Stop on convergence instead of a fixed iteration count
* [ ] Splitting the data into train/test sets
