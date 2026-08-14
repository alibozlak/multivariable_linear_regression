
pub struct WithoutFeatureScaling {
    real_inputs : Vec<Vec<f64>>,
    real_outputs : Vec<f64>,
    m : usize,
    n : usize,
    coefficients : Vec<f64>
}

impl WithoutFeatureScaling {
    pub fn new(
        real_inputs : Vec<Vec<f64>>,
        real_outputs : Vec<f64>,
        f_initial_coefficients : Vec<f64>
    ) -> WithoutFeatureScaling {
        let (real_inputs, real_outputs, m, n, initial_coefficients)
            = Self::validate_data_set(real_inputs, real_outputs, f_initial_coefficients);
        Self { real_inputs, real_outputs, m, n, coefficients: initial_coefficients }
    }

    pub fn train_model(&mut self, learning_rate : f64, loop_count : usize) -> Vec<f64> {
        let mut temp_coefficients  = vec![0.0; self.n + 1];
        for i in 0..loop_count {
            for j in 0..self.n {
                temp_coefficients[j] = self.coefficients[j] - learning_rate * self.dJ_daj(j);
            }

            temp_coefficients[self.n] = self.coefficients[self.n] - learning_rate * self.dJ_db();

            for j in 0..(self.n + 1) {
                self.coefficients[j] = temp_coefficients[j];
            }
        }

        self.coefficients.clone()
    }

    fn dJ_daj(&self, j : usize) -> f64 {
        let mut result : f64 = 0.0;
        for i in 0..self.m {
            result += self.real_inputs[i][j] * (self.f(i) - self.real_outputs[i]);
        }

        result * 2. / (self.m as f64)
    }

    fn dJ_db(&self) -> f64 {
        let mut result : f64 = 0.0;
        for i in 0..self.m {
            result += (self.f(i) - self.real_outputs[i]);
        }

        result * 2. / (self.m as f64)
    }

    pub fn J(&self) -> f64 {
        let mut result : f64 = 0.0;
        for i in 0..self.m {
            result += (self.f(i) - self.real_outputs[i]).powi(2);
        }

        result / (self.m as f64)
    }

    fn f(&self, m : usize) -> f64 {
        let mut result : f64 = 0.0;
        if self.n < 16 {
            for i in 0..self.n {
                result += self.coefficients[i] * self.real_inputs[m][i];
            }
        } else {
            // FixMe : Vectorization
            for i in 0..self.n {
                result += self.coefficients[i] * self.real_inputs[m][i];
            }
        }

        result + self.coefficients[self.n]
    }

    fn validate_data_set(
        real_inputs : Vec<Vec<f64>>,
        real_outputs : Vec<f64>,
        initial_coefficients : Vec<f64>
    )
        -> (Vec<Vec<f64>>, Vec<f64>, usize, usize, Vec<f64>)
    {
        let m : usize = real_inputs.len();
        if m != real_outputs.len() {
            panic!("Real Inputs and Output Sample size (m) not equal !!");
        }

        let n : usize = real_inputs[0].len();

        if n + 1 != initial_coefficients.len() {
            panic!("Feature count (n) and a real input sample size (n) not equal !!");
        }

        (real_inputs, real_outputs, m, n, initial_coefficients)
    }

}