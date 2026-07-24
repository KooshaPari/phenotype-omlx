use crate::backend::{Backend, BackendError, Completion, Likelihood};
use pyo3::prelude::*;
use pyo3::types::PyDict;

pub(crate) struct PythonBackendWrapper {
    pub obj: Py<PyAny>,
}

impl Backend for PythonBackendWrapper {
    fn complete(
        &self,
        prompt: &str,
        max_tokens: usize,
    ) -> std::result::Result<Completion, BackendError> {
        Python::attach(|py| {
            let py_obj = self.obj.bind(py);
            let res = if py_obj.hasattr("complete").unwrap_or(false) {
                py_obj.call_method1("complete", (prompt, max_tokens))
            } else if py_obj.is_callable() {
                py_obj.call1((prompt, max_tokens))
            } else {
                return Err(BackendError::InvalidResponse {
                    message: "Backend object has no 'complete' method and is not callable".into(),
                });
            };

            let res = res.map_err(|e| BackendError::Unavailable {
                message: e.to_string(),
            })?;

            if let Ok(text) = res.extract::<String>() {
                let prompt_tokens = prompt.split_whitespace().count();
                let completion_tokens = text.split_whitespace().count();
                Ok(Completion {
                    text,
                    prompt_tokens,
                    completion_tokens,
                    latency_ms: 0.0,
                })
            } else if let Ok(dict) = res.cast::<PyDict>() {
                let text: String = match dict.get_item("text") {
                    Ok(Some(val)) => {
                        val.extract()
                            .map_err(|e: pyo3::PyErr| BackendError::InvalidResponse {
                                message: e.to_string(),
                            })?
                    }
                    _ => {
                        return Err(BackendError::InvalidResponse {
                            message: "missing 'text'".into(),
                        })
                    }
                };

                let prompt_tokens: usize = dict
                    .get_item("prompt_tokens")
                    .ok()
                    .flatten()
                    .and_then(|v| v.extract().ok())
                    .unwrap_or_else(|| prompt.split_whitespace().count());

                let completion_tokens: usize = dict
                    .get_item("completion_tokens")
                    .ok()
                    .flatten()
                    .and_then(|v| v.extract().ok())
                    .unwrap_or_else(|| text.split_whitespace().count());

                let latency_ms: f64 = dict
                    .get_item("latency_ms")
                    .ok()
                    .flatten()
                    .and_then(|v| v.extract().ok())
                    .unwrap_or(0.0);

                Ok(Completion {
                    text,
                    prompt_tokens,
                    completion_tokens,
                    latency_ms,
                })
            } else if let Ok((text, prompt_tokens, completion_tokens, latency_ms)) =
                res.extract::<(String, usize, usize, f64)>()
            {
                Ok(Completion {
                    text,
                    prompt_tokens,
                    completion_tokens,
                    latency_ms,
                })
            } else {
                Err(BackendError::InvalidResponse {
                    message: format!("unexpected return type from backend complete: {res}"),
                })
            }
        })
    }

    fn log_likelihood(
        &self,
        prompt: &str,
        continuation: &str,
    ) -> std::result::Result<Likelihood, BackendError> {
        Python::attach(|py| {
            let py_obj = self.obj.bind(py);
            let res = if py_obj.hasattr("log_likelihood").unwrap_or(false) {
                py_obj.call_method1("log_likelihood", (prompt, continuation))
            } else if py_obj.hasattr("log_prob").unwrap_or(false) {
                py_obj.call_method1("log_prob", (prompt, continuation))
            } else {
                let comp = self.complete(prompt, continuation.split_whitespace().count().max(1))?;
                let log_prob = if comp.text.trim() == continuation.trim() {
                    0.0
                } else {
                    -10.0
                };
                return Ok(Likelihood {
                    log_probability: log_prob,
                    token_count: comp.completion_tokens,
                    latency_ms: comp.latency_ms,
                });
            };

            let res = res.map_err(|e| BackendError::Unavailable {
                message: e.to_string(),
            })?;

            if let Ok(log_probability) = res.extract::<f64>() {
                let token_count = continuation.split_whitespace().count();
                Ok(Likelihood {
                    log_probability,
                    token_count,
                    latency_ms: 0.0,
                })
            } else if let Ok(dict) = res.cast::<PyDict>() {
                let prob_val = match dict.get_item("log_probability") {
                    Ok(Some(val)) => Some(val),
                    _ => match dict.get_item("log_prob") {
                        Ok(Some(val)) => Some(val),
                        _ => None,
                    },
                };
                let log_probability: f64 = match prob_val {
                    Some(val) => {
                        val.extract()
                            .map_err(|e: pyo3::PyErr| BackendError::InvalidResponse {
                                message: e.to_string(),
                            })?
                    }
                    None => {
                        return Err(BackendError::InvalidResponse {
                            message: "missing 'log_probability'".into(),
                        })
                    }
                };

                let token_count: usize = dict
                    .get_item("token_count")
                    .ok()
                    .flatten()
                    .and_then(|v| v.extract().ok())
                    .unwrap_or_else(|| continuation.split_whitespace().count());

                let latency_ms: f64 = dict
                    .get_item("latency_ms")
                    .ok()
                    .flatten()
                    .and_then(|v| v.extract().ok())
                    .unwrap_or(0.0);

                Ok(Likelihood {
                    log_probability,
                    token_count,
                    latency_ms,
                })
            } else if let Ok((log_probability, token_count, latency_ms)) =
                res.extract::<(f64, usize, f64)>()
            {
                Ok(Likelihood {
                    log_probability,
                    token_count,
                    latency_ms,
                })
            } else {
                Err(BackendError::InvalidResponse {
                    message: format!("unexpected return type from backend log_likelihood: {res}"),
                })
            }
        })
    }
}
