#![allow(non_snake_case)]

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use thiserror::Error;

type Float = f64;

#[derive(Error, Debug)]
enum KalmanError {
    #[error("failed to invert scalar {scalar_name} in operation")]
    FailedScalarInverse { scalar_name: &'static str },
}
type KalmanResult<T> = std::result::Result<T, KalmanError>;

impl From<KalmanError> for PyErr {
    fn from(k: KalmanError) -> Self {
        PyValueError::new_err(k.to_string())
    }
}

#[derive(Debug, Clone)]
#[pyclass]
struct ScalarKalman {
    x: Float,
    P: Float,
    A: Float,
    H: Float,
    Q: Float,
    R: Float,
}

impl ScalarKalman {
    fn new(A: Float, H: Float, Q: Float, R: Float, x0: Option<Float>, P0: Option<Float>) -> Self {
        let x = if let Some(x0) = x0 { x0 } else { 0.0 };
        let P = if let Some(P0) = P0 { P0 } else { 0.0 };
        Self { x, P, A, H, Q, R }
    }

    fn predict(&mut self) {
        self.x *= self.A;
        self.P = self.A * self.P * self.A + self.Q;
    }

    fn update(&mut self, z: Float) -> KalmanResult<()> {
        let y = z - self.H * self.x;
        let S = self.H * self.P * self.H + self.R;
        if S.abs() < 1e-8 {
            return Err(KalmanError::FailedScalarInverse {
                scalar_name: "Innovation (measurement pre-fit residual `S`)",
            });
        }
        let S_inv = 1. / S;
        let K = self.P * self.H * S_inv;
        self.x += K * y;
        self.P *= 1.0 - K * self.H;
        Ok(())
    }

    fn advance(&mut self, z: Float) -> KalmanResult<Float> {
        self.predict();
        self.update(z)?;
        Ok(self.x)
    }
}

#[pymethods]
impl ScalarKalman {
    #[new]
    fn py_new(
        A: Float,
        H: Float,
        Q: Float,
        R: Float,
        x0: Option<Float>,
        P0: Option<Float>,
    ) -> Self {
        Self::new(A, H, Q, R, x0, P0)
    }
    #[pyo3(name = "advance")]
    fn py_advance(&mut self, z: Float) -> PyResult<Float> {
        Ok(self.advance(z)?)
    }
}

#[pyfunction]
fn kfilter(filter: &mut ScalarKalman, vec: Vec<Float>) -> PyResult<Vec<Float>> {
    let mut out = Vec::with_capacity(vec.len());
    for &v in vec.iter() {
        out.push(filter.advance(v)?)
    }
    Ok(out)
}

/// A Python module implemented in Rust.
#[pymodule]
fn kalman_no_control(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<ScalarKalman>()?;
    m.add_function(wrap_pyfunction!(kfilter, m)?)?;
    Ok(())
}
