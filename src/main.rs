mod json_data_set;
mod without_feature_scaling;

fn main() {
    let gemini_created_data_set = json_data_set::load(json_data_set::DEFAULT_PATH);

    let learning_rate_alpha = 0.00003;
    let loop_count_for_train = 1_000_000;

    let n = without_feature_scaling::n(&gemini_created_data_set);
    let mut coefficients_array = vec![0.0; n];
    let mut constant_coefficient = 0.0;

    without_feature_scaling::train_data_set(
        &mut coefficients_array,
        &mut constant_coefficient,
        &gemini_created_data_set,
        loop_count_for_train,
        learning_rate_alpha,
    );

    // **** Learning Too Slow !! *****
}
